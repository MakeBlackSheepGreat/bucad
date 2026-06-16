"""FastAPI server exposing the BUCAD inference workflow to a separate frontend."""

from __future__ import annotations

import argparse
import base64
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components.localization import (
    localize_confidence_band,
    localize_disclaimer,
    localize_final_label,
    localize_recommendation,
    localize_status,
    localize_warning,
)
from backend.agent_harness import (
    AgentInterpretationRequest,
    AgentInterpretationResponse,
    BreastTumorAgentHarness,
)
from backend.knowledge import KnowledgeEntry, list_knowledge_entries, search_knowledge
from backend.patients import PatientCase, PatientCaseCreate, PatientCaseRepository
from src.engine.errors import InvalidInputError
from src.engine.inference import BreastUltrasoundInferenceService
from src.preprocess.io import SUPPORTED_SUFFIXES, cv2, ensure_three_channels
from src.utils.runtime import is_torch_available
from src.utils.results import AUXILIARY_USE_DISCLAIMER, InferenceResponse


DEFAULT_CONFIG_PATH = Path("configs/inference/demo.yml")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from a local UTF-8 env file."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_file(Path(__file__).resolve().parents[1] / ".env")


class ThresholdInfo(BaseModel):
    """Threshold slider contract shared with the Vue frontend."""

    min: float = 0.1
    max: float = 0.9
    step: float = 0.001
    default: float


class RuntimeInfo(BaseModel):
    """Public runtime metadata for initial frontend rendering."""

    app_name: str = "乳腺超声肿瘤良恶性分类辅助诊断系统（BUCAD）"
    ensemble_display_name: str
    model_identifier: str
    threshold: ThresholdInfo
    segmentation_enabled: bool
    gradcam_enabled: bool
    classifier_member_count: int
    disclaimer: str


class DiagnosticResultPayload(BaseModel):
    """JSON-safe diagnosis result returned by the API."""

    benign_probability: float
    malignant_probability: float
    final_label: str
    final_label_text: str
    confidence_band: str
    confidence_band_text: str
    recommendation_text: str
    analysis_timestamp: str
    model_version: str
    model_display_name: str
    auxiliary_use_disclaimer: str


class ImagePayload(BaseModel):
    """Base64 PNG visual outputs for browser rendering."""

    original: str | None = None
    lesion: str | None = None
    explanation: str | None = None


class DiagnosisPayload(BaseModel):
    """Top-level API response for a single uploaded image."""

    status: str
    status_text: str
    input_filename: str
    result: DiagnosticResultPayload | None
    images: ImagePayload = Field(default_factory=ImagePayload)
    warnings: list[str] = Field(default_factory=list)
    raw_warnings: list[str] = Field(default_factory=list)
    lesion_visualization_missing_reason: str | None = None
    explanation_missing_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _config_path_from_env() -> str:
    """Return the configured inference YAML path."""
    return os.environ.get("BUCAD_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))


@lru_cache(maxsize=4)
def _service_from_config_path(config_path: str) -> BreastUltrasoundInferenceService:
    """Build and cache inference service instances by config path."""
    return BreastUltrasoundInferenceService.from_config(config_path)


def _default_service_factory() -> BreastUltrasoundInferenceService:
    """Default dependency used by production routes."""
    return _service_from_config_path(_config_path_from_env())


def _cors_origins_from_env() -> list[str]:
    """Parse allowed CORS origins for local frontend development."""
    raw = os.environ.get(
        "BUCAD_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def _patient_repository() -> PatientCaseRepository:
    """Return the local patient case repository."""
    return PatientCaseRepository()


@lru_cache(maxsize=1)
def _agent_harness() -> BreastTumorAgentHarness:
    """Return the local medical interpretation agent harness."""
    return BreastTumorAgentHarness()


def _runtime_display_name(service: BreastUltrasoundInferenceService) -> str:
    """Return the configured model display name."""
    return str(
        service.runtime_config.get(
            "ensemble_display_name",
            "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate",
        )
    )


def _runtime_default_threshold(service: BreastUltrasoundInferenceService) -> float:
    """Return the configured default diagnosis threshold."""
    return float(service.runtime_config.get("default_threshold", 0.51))


def _read_upload_as_grayscale(raw: bytes, filename: str) -> np.ndarray:
    """Decode an uploaded browser file into the grayscale array expected by inference."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise InvalidInputError(f"Unsupported image file type: {suffix or 'unknown'}")
    if not raw:
        raise InvalidInputError("Image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InvalidInputError("Uploaded image is larger than 25MB.")
    if cv2 is None:
        raise InvalidInputError("No image backend is installed. Install opencv-python or Pillow.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise InvalidInputError(f"Unable to read image: {filename}")
    return image


def _array_to_png_data_url(image: Any | None) -> str | None:
    """Convert a NumPy image array into a browser-ready data URL."""
    if image is None:
        return None
    if cv2 is None:
        return None
    array = np.asarray(ensure_three_channels(image))
    if array.dtype != np.uint8:
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 3:
        encoded_input = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    else:
        encoded_input = array
    ok, encoded = cv2.imencode(".png", encoded_input)
    if not ok:
        return None
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _model_display_name(model_version: str, metadata: dict[str, Any]) -> str:
    """Prefer the configured display name in API responses."""
    configured = metadata.get("ensemble_display_name")
    if configured:
        return str(configured)
    if "convnext_tiny" in model_version and "tf_efficientnetv2_s" in model_version:
        return "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate"
    return model_version


def _serialize_diagnosis(response: InferenceResponse) -> DiagnosisPayload:
    """Translate the internal inference response into the public JSON contract."""
    metadata = dict(response.metadata or {})
    result_payload = None
    if response.result is not None:
        result = response.result
        result_payload = DiagnosticResultPayload(
            benign_probability=result.benign_probability,
            malignant_probability=result.malignant_probability,
            final_label=result.final_label,
            final_label_text=localize_final_label(result.final_label),
            confidence_band=result.confidence_band,
            confidence_band_text=localize_confidence_band(result.confidence_band),
            recommendation_text=localize_recommendation(
                result.final_label,
                result.confidence_band,
            ),
            analysis_timestamp=result.analysis_timestamp,
            model_version=result.model_version,
            model_display_name=_model_display_name(result.model_version, metadata),
            auxiliary_use_disclaimer=localize_disclaimer(result.auxiliary_use_disclaimer),
        )
    return DiagnosisPayload(
        status=response.status,
        status_text=localize_status(response.status),
        input_filename=response.input_filename,
        result=result_payload,
        images=ImagePayload(
            original=_array_to_png_data_url(response.original_image_view),
            lesion=_array_to_png_data_url(response.lesion_overlay_view),
            explanation=_array_to_png_data_url(response.explanation_view),
        ),
        warnings=[localize_warning(warning) for warning in response.warnings],
        raw_warnings=list(response.warnings),
        lesion_visualization_missing_reason=(
            localize_warning(response.lesion_visualization_missing_reason)
            if response.lesion_visualization_missing_reason
            else None
        ),
        explanation_missing_reason=(
            localize_warning(response.explanation_missing_reason)
            if response.explanation_missing_reason
            else None
        ),
        metadata=metadata,
    )


def create_app(
    *,
    service_factory: Callable[[], BreastUltrasoundInferenceService] | None = None,
    patient_repository_factory: Callable[[], PatientCaseRepository] | None = None,
    agent_harness_factory: Callable[[], BreastTumorAgentHarness] | None = None,
) -> FastAPI:
    """Create the FastAPI application with an injectable service for tests."""
    resolve_service = service_factory or _default_service_factory
    resolve_patient_repository = patient_repository_factory or _patient_repository
    resolve_agent_harness = agent_harness_factory or _agent_harness
    api = FastAPI(
        title="BUCAD Inference API",
        version="1.0.0",
        description="Separated HTTP API for the BUCAD single-image diagnosis workflow.",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        """Simple liveness endpoint."""
        return {"status": "ok"}

    @api.get("/ready")
    def ready() -> dict[str, Any]:
        """Readiness endpoint covering API and patient database availability."""
        repository = resolve_patient_repository()
        return {
            "status": "ok",
            "patient_database": str(repository.db_path),
            "python_executable": sys.executable,
            "torch_available": is_torch_available(),
        }

    @api.get("/api/runtime", response_model=RuntimeInfo)
    def runtime_info() -> RuntimeInfo:
        """Return metadata needed to initialize the standalone Vue frontend."""
        service = resolve_service()
        return RuntimeInfo(
            ensemble_display_name=_runtime_display_name(service),
            model_identifier=service._model_identifier(),
            threshold=ThresholdInfo(default=_runtime_default_threshold(service)),
            segmentation_enabled=bool(service.runtime_config.get("segmentation_enabled", True)),
            gradcam_enabled=bool(service.runtime_config.get("gradcam_enabled", True)),
            classifier_member_count=len(service._resolved_classifier_member_configs()),
            disclaimer=localize_disclaimer(AUXILIARY_USE_DISCLAIMER),
        )

    @api.post("/api/diagnose", response_model=DiagnosisPayload)
    async def diagnose(
        file: UploadFile = File(...),
        threshold: float = Form(...),
        need_segmentation: bool = Form(True),
        need_explanation: bool = Form(True),
    ) -> DiagnosisPayload:
        """Run single-image diagnosis from a browser multipart upload."""
        filename = file.filename or "uploaded.png"
        try:
            image = _read_upload_as_grayscale(await file.read(), filename)
        except InvalidInputError as exc:
            raise HTTPException(status_code=400, detail=localize_warning(str(exc))) from exc
        response = resolve_service().diagnose(
            image,
            input_filename=filename,
            decision_threshold=threshold,
            need_segmentation=need_segmentation,
            need_explanation=need_explanation,
        )
        return _serialize_diagnosis(response)

    @api.get("/api/patients", response_model=list[PatientCase])
    def list_patient_cases(q: str | None = None) -> list[PatientCase]:
        """List imported patient cases."""
        return resolve_patient_repository().list_cases(q)

    @api.post("/api/patients", response_model=PatientCase, status_code=201)
    def create_patient_case(payload: PatientCaseCreate) -> PatientCase:
        """Import a new patient case into the local database."""
        return resolve_patient_repository().create_case(payload)

    @api.post("/api/patients/seed-samples", response_model=list[PatientCase])
    def seed_sample_patient_cases() -> list[PatientCase]:
        """Insert bundled demo patient cases into the local database."""
        return resolve_patient_repository().seed_sample_cases()

    @api.get("/api/patients/{case_id}", response_model=PatientCase)
    def get_patient_case(case_id: int) -> PatientCase:
        """Return one patient case."""
        case = resolve_patient_repository().get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="病例不存在。")
        return case

    @api.get("/api/knowledge", response_model=list[KnowledgeEntry])
    def list_knowledge(q: str | None = None, category: str | None = None) -> list[KnowledgeEntry]:
        """Return curated breast tumor knowledge entries."""
        if q:
            return search_knowledge(q, limit=8)
        return list_knowledge_entries(category)

    @api.post("/api/agent/interpret", response_model=AgentInterpretationResponse)
    def interpret_with_agent(payload: AgentInterpretationRequest) -> AgentInterpretationResponse:
        """Generate a structured medical interpretation using knowledge retrieval."""
        return resolve_agent_harness().interpret(payload)

    @api.post("/api/agent/interpret/stream")
    def stream_interpret_with_agent(payload: AgentInterpretationRequest) -> StreamingResponse:
        """Stream a structured medical interpretation using knowledge retrieval."""
        return StreamingResponse(
            resolve_agent_harness().interpret_stream(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return api


app = create_app()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the standalone API server."""
    parser = argparse.ArgumentParser(description="Launch the BUCAD FastAPI backend.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument("--config", type=Path, default=None, help="Inference config path.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    return parser.parse_args()


def main() -> None:
    """Launch the HTTP API server."""
    args = parse_args()
    if args.config is not None:
        os.environ["BUCAD_CONFIG_PATH"] = str(args.config)
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
