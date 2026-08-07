"""Unit tests for the separated FastAPI backend contract."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.patients import PatientCaseRepository
from backend.server import _patient_repository, create_app
from src.preprocess.io import cv2
from src.utils.results import InferenceResponse, build_diagnostic_result


class StubInferenceService:
    """Small inference service double that avoids loading model checkpoints."""

    runtime_config = {
        "default_threshold": 0.51,
        "ensemble_display_name": "Stub Ensemble",
        "segmentation_enabled": True,
        "gradcam_enabled": True,
    }

    def _model_identifier(self) -> str:
        """Return the stable identifier exposed by the stub runtime."""
        return "stub-model"

    def _resolved_classifier_member_configs(self) -> list[dict[str, str]]:
        """Return one lightweight member for frontend metadata tests."""
        return [{"model": "stub", "checkpoint": "stub.pt"}]

    def diagnose(
        self,
        image_input,
        *,
        input_filename: str | None = None,
        decision_threshold: float | None = None,
        need_segmentation: bool = True,
        need_explanation: bool = True,
    ) -> InferenceResponse:
        """Return a deterministic completed response without model loading."""
        del decision_threshold, need_segmentation, need_explanation
        image = np.asarray(image_input, dtype=np.uint8)
        visual = np.repeat(image[..., None], 3, axis=2)
        return InferenceResponse(
            status="completed",
            input_filename=input_filename or "uploaded.png",
            result=build_diagnostic_result(0.22, 0.78, threshold=0.51, model_version="stub-model"),
            original_image_view=visual,
            lesion_overlay_view=visual,
            explanation_view=visual,
            metadata={
                "decision_threshold": 0.51,
                "ensemble_display_name": "Stub Ensemble",
            },
        )


def _png_bytes() -> bytes:
    """Encode a small grayscale PNG used by upload endpoint tests."""
    image = np.full((40, 40), 96, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _client_with_patient_database(tmp_path) -> TestClient:
    """Create an isolated API client backed by a temporary SQLite database."""
    repository = PatientCaseRepository(tmp_path / "patients.db")
    return TestClient(
        create_app(
            service_factory=StubInferenceService,
            patient_repository_factory=lambda: repository,
        )
    )


def _patient_payload(patient_code: str = "BUCAD-2026-001") -> dict[str, object]:
    """Return a valid patient record payload for backend endpoint tests."""
    return {
        "patient_code": patient_code,
        "name": "张三",
        "sex": "女",
        "age": 45,
        "contact": "13800000000",
        "visit_date": "2026-06-16",
        "department": "乳腺外科",
        "primary_complaint": "右乳触及肿块 2 周。",
        "ultrasound_description": "右乳低回声结节，边界欠清。",
        "birads_category": "4A",
        "pathology_status": "待完善",
        "risk_level": "中风险",
        "notes": "建议结合穿刺活检。",
    }


def test_runtime_endpoint_returns_frontend_metadata() -> None:
    """Verify the runtime endpoint exposes frontend initialization metadata."""
    client = TestClient(create_app(service_factory=StubInferenceService))

    response = client.get("/api/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ensemble_display_name"] == "Stub Ensemble"
    assert payload["threshold"]["default"] == 0.51
    assert payload["classifier_member_count"] == 1


def test_ready_endpoint_reports_runtime_environment(tmp_path) -> None:
    """Verify readiness includes runtime and database diagnostics."""
    client = _client_with_patient_database(tmp_path)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["patient_database"].endswith("patients.db")
    assert payload["python_executable"]
    assert isinstance(payload["torch_available"], bool)


def test_diagnose_endpoint_returns_json_safe_visuals() -> None:
    """Verify image diagnosis serializes browser-ready visual data URLs."""
    client = TestClient(create_app(service_factory=StubInferenceService))

    response = client.post(
        "/api/diagnose",
        data={
            "threshold": "0.51",
            "need_segmentation": "true",
            "need_explanation": "true",
        },
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result"]["final_label_text"] == "恶性"
    assert payload["images"]["original"].startswith("data:image/png;base64,")


def test_patient_case_create_list_and_detail(tmp_path) -> None:
    """Verify a patient case can be created, listed, and retrieved."""
    client = _client_with_patient_database(tmp_path)

    created_response = client.post("/api/patients", json=_patient_payload())

    assert created_response.status_code == 201
    created = created_response.json()
    assert created["id"] >= 1
    assert created["patient_code"] == "BUCAD-2026-001"
    assert created["name"] == "张三"

    list_response = client.get("/api/patients")
    assert list_response.status_code == 200
    cases = list_response.json()
    assert len(cases) == 1
    assert cases[0]["id"] == created["id"]

    detail_response = client.get(f"/api/patients/{created['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["ultrasound_description"] == "右乳低回声结节，边界欠清。"


def test_patient_case_search_filters_records(tmp_path) -> None:
    """Verify patient search limits results to matching records."""
    client = _client_with_patient_database(tmp_path)
    assert client.post("/api/patients", json=_patient_payload("BUCAD-2026-001")).status_code == 201
    second = _patient_payload("BUCAD-2026-002")
    second["name"] = "李四"
    second["risk_level"] = "高风险"
    assert client.post("/api/patients", json=second).status_code == 201

    response = client.get("/api/patients", params={"q": "高风险"})

    assert response.status_code == 200
    cases = response.json()
    assert len(cases) == 1
    assert cases[0]["patient_code"] == "BUCAD-2026-002"


def test_seed_sample_patient_cases_inserts_demo_records(tmp_path) -> None:
    """Verify sample insertion is populated and idempotent."""
    client = _client_with_patient_database(tmp_path)

    response = client.post("/api/patients/seed-samples")

    assert response.status_code == 200
    cases = response.json()
    assert len(cases) >= 6
    assert {"张三", "李四", "王芳", "赵敏", "陈静", "周华"}.issubset(
        {case["name"] for case in cases}
    )

    repeat_response = client.post("/api/patients/seed-samples")
    assert repeat_response.status_code == 200
    repeat_cases = repeat_response.json()
    assert len(repeat_cases) == len(cases)
    assert {"DEMO-ZS-001", "DEMO-LS-002", "DEMO-WF-003", "DEMO-ZM-004"}.issubset(
        {case["patient_code"] for case in repeat_cases}
    )


def test_patient_repository_preloads_demo_cases_when_requested(tmp_path) -> None:
    """Verify startup seeding keeps bundled cases available without duplicates."""
    database_path = tmp_path / "patients.db"

    first_repository = PatientCaseRepository(database_path, seed_samples=True)
    first_cases = first_repository.list_cases()
    second_repository = PatientCaseRepository(database_path, seed_samples=True)
    second_cases = second_repository.list_cases()

    assert len(first_cases) == 6
    assert len(second_cases) == 6
    assert {case.patient_code for case in second_cases} == {
        "DEMO-ZS-001",
        "DEMO-LS-002",
        "DEMO-WF-003",
        "DEMO-ZM-004",
        "DEMO-CJ-005",
        "DEMO-ZH-006",
    }


def test_default_api_preloads_demo_patient_cases(tmp_path, monkeypatch) -> None:
    """Verify the production repository provides demo cases on first API request."""
    monkeypatch.setenv("BUCAD_PATIENT_DB_PATH", str(tmp_path / "default-patients.db"))
    _patient_repository.cache_clear()
    try:
        client = TestClient(create_app(service_factory=StubInferenceService))
        response = client.get("/api/patients")
    finally:
        _patient_repository.cache_clear()

    assert response.status_code == 200
    assert len(response.json()) == 6


def test_patient_case_not_found_returns_404(tmp_path) -> None:
    """Verify an unknown patient identifier produces a 404 response."""
    client = _client_with_patient_database(tmp_path)

    response = client.get("/api/patients/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "病例不存在。"


def test_knowledge_endpoint_returns_curated_entries() -> None:
    """Verify knowledge search returns relevant curated entries."""
    client = TestClient(create_app(service_factory=StubInferenceService))

    response = client.get("/api/knowledge", params={"q": "HER2 靶向治疗"})

    assert response.status_code == 200
    entries = response.json()
    assert entries
    assert any("HER2" in entry["title"] or "HER2" in entry["summary"] for entry in entries)


def test_agent_endpoint_returns_structured_fallback_without_api_key(tmp_path, monkeypatch) -> None:
    """Verify the non-streaming agent route redacts local fallback context."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = _client_with_patient_database(tmp_path)
    created = client.post("/api/patients", json=_patient_payload()).json()

    response = client.post(
        "/api/agent/interpret",
        json={
            "question": "请解释为什么需要穿刺活检，并提示还缺哪些病理信息。",
            "patient_case": created,
            "diagnosis": {
                "result": {
                    "final_label_text": "恶性",
                    "malignant_probability": 0.78,
                    "benign_probability": 0.22,
                    "confidence_band_text": "高",
                    "model_display_name": "Stub Ensemble",
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is True
    assert payload["provider"] == "local-fallback"
    assert "1. 结论摘要" in payload["answer"]
    assert "知识库依据" not in payload["answer"]
    assert "已检索" not in payload["answer"]
    assert "张三" not in payload["answer"]
    assert "BUCAD-2026-001" not in payload["answer"]
    assert "13800000000" not in payload["answer"]
    assert payload["retrieved_knowledge"]


def test_agent_stream_endpoint_returns_sse_fallback_without_api_key(tmp_path, monkeypatch) -> None:
    """Verify the streaming agent route emits a redacted fallback event."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = _client_with_patient_database(tmp_path)
    created = client.post("/api/patients", json=_patient_payload()).json()

    response = client.post(
        "/api/agent/interpret/stream",
        json={
            "question": "请解释为什么需要穿刺活检。",
            "patient_case": created,
            "diagnosis": None,
        },
    )

    assert response.status_code == 200
    assert "event: metadata" in response.text
    assert "event: fallback" in response.text
    assert "1. 结论摘要" in response.text
    assert "知识库依据" not in response.text
    assert "已检索" not in response.text
    assert "张三" not in response.text
    assert "BUCAD-2026-001" not in response.text
    assert "13800000000" not in response.text
    assert "retrieved_knowledge" in response.text
