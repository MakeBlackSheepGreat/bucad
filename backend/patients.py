"""SQLite-backed patient case repository for the local BUCAD platform."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_PATIENT_DB_PATH = Path("data/patients.db")


class PatientCaseCreate(BaseModel):
    """Payload accepted when importing a patient case."""

    patient_code: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=80)
    sex: str = Field("未填写", max_length=16)
    age: int | None = Field(default=None, ge=0, le=130)
    contact: str = Field("", max_length=80)
    visit_date: str = Field("", max_length=32)
    department: str = Field("乳腺外科", max_length=80)
    primary_complaint: str = Field("", max_length=800)
    ultrasound_description: str = Field("", max_length=2000)
    birads_category: str = Field("", max_length=32)
    pathology_status: str = Field("待完善", max_length=80)
    risk_level: str = Field("待评估", max_length=32)
    notes: str = Field("", max_length=2000)


class PatientCase(PatientCaseCreate):
    """Stored patient case returned by the API."""

    id: int
    created_at: str
    updated_at: str


SAMPLE_PATIENT_CASES: tuple[PatientCaseCreate, ...] = (
    PatientCaseCreate(
        patient_code="DEMO-ZS-001",
        name="张三",
        sex="女",
        age=46,
        contact="示例数据",
        visit_date="2026-06-16",
        department="乳腺外科",
        primary_complaint="右乳外上象限触及无痛性肿块 2 周。",
        ultrasound_description=(
            "右乳 10 点方向低回声结节，约 18 mm x 12 mm，形态欠规则，边界欠清，"
            "纵横比接近 1，后方回声轻度衰减，CDFI 可见少量血流信号。"
        ),
        birads_category="4A",
        pathology_status="待完善",
        risk_level="中风险",
        notes="示例病例：建议结合钼靶或 MRI，并评估粗针穿刺活检。非真实患者资料。",
    ),
    PatientCaseCreate(
        patient_code="DEMO-LS-002",
        name="李四",
        sex="女",
        age=58,
        contact="示例数据",
        visit_date="2026-06-16",
        department="乳腺外科",
        primary_complaint="左乳肿块逐渐增大 1 月，偶有乳头牵拉感。",
        ultrasound_description=(
            "左乳 2 点方向不规则低回声肿块，约 26 mm x 19 mm，边界不清，"
            "呈非平行生长，后方声影明显，腋窝可见皮质增厚淋巴结。"
        ),
        birads_category="4C",
        pathology_status="待完善",
        risk_level="高风险",
        notes="示例病例：优先建议组织学取材，并补充 ER、PR、HER2、Ki-67 与腋窝评估。非真实患者资料。",
    ),
    PatientCaseCreate(
        patient_code="DEMO-WF-003",
        name="王芳",
        sex="女",
        age=34,
        contact="示例数据",
        visit_date="2026-06-16",
        department="乳腺外科",
        primary_complaint="左乳外上象限体检发现结节 3 月，无明显疼痛。",
        ultrasound_description=(
            "左乳 1 点方向椭圆形低回声结节，约 13 mm x 7 mm，边界清楚，"
            "平行生长，内部回声尚均匀，后方回声无明显衰减，CDFI 未见丰富血流。"
        ),
        birads_category="3",
        pathology_status="随访中",
        risk_level="低风险",
        notes="示例病例：倾向良性实性结节，建议结合既往影像进行短期随访。非真实患者资料。",
    ),
    PatientCaseCreate(
        patient_code="DEMO-ZM-004",
        name="赵敏",
        sex="女",
        age=62,
        contact="示例数据",
        visit_date="2026-06-16",
        department="乳腺外科",
        primary_complaint="右乳癌保乳术后 2 年复查，局部偶有牵拉不适。",
        ultrasound_description=(
            "右乳术区可见条索样低回声及局部结构改变，未见明确新发占位；"
            "腋窝未见形态明显异常淋巴结。"
        ),
        birads_category="2",
        pathology_status="既往恶性，术后随访",
        risk_level="低风险",
        notes="示例病例：需保留既往治疗资料和影像对比，区分术后瘢痕、脂肪坏死与复发。非真实患者资料。",
    ),
    PatientCaseCreate(
        patient_code="DEMO-CJ-005",
        name="陈静",
        sex="女",
        age=41,
        contact="示例数据",
        visit_date="2026-06-16",
        department="乳腺外科",
        primary_complaint="右乳头间断性血性溢液 1 月。",
        ultrasound_description=(
            "右乳乳晕区导管扩张，导管内可见约 8 mm 实性低回声结节，"
            "内部可见点状血流信号，边界尚清，未见明确后方声影。"
        ),
        birads_category="4B",
        pathology_status="待完善",
        risk_level="中高风险",
        notes="示例病例：考虑导管内乳头状病变等可能，需结合钼靶、导管相关检查和组织学取材。非真实患者资料。",
    ),
    PatientCaseCreate(
        patient_code="DEMO-ZH-006",
        name="周华",
        sex="女",
        age=52,
        contact="示例数据",
        visit_date="2026-06-16",
        department="乳腺外科",
        primary_complaint="左乳皮肤红肿增厚 3 周，抗炎治疗后改善不明显。",
        ultrasound_description=(
            "左乳皮肤及皮下组织增厚，乳腺实质回声紊乱，局部可见不规则低回声区，"
            "边界欠清，腋窝可见皮质增厚淋巴结。"
        ),
        birads_category="5",
        pathology_status="待完善",
        risk_level="高风险",
        notes="示例病例：需警惕炎性乳腺癌或局部进展性病变，建议尽快组织学确认及分期评估。非真实患者资料。",
    ),
)


def patient_database_path_from_env() -> Path:
    """Return the configured patient database path."""
    return Path(os.environ.get("BUCAD_PATIENT_DB_PATH", str(DEFAULT_PATIENT_DB_PATH)))


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a Pydantic model dictionary across Pydantic v1/v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class PatientCaseRepository:
    """Small repository around SQLite patient case records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Create a repository and initialize the schema if needed."""
        self.db_path = Path(db_path) if db_path is not None else patient_database_path_from_env()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with dict-like rows."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        """Create the patient case table."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    sex TEXT NOT NULL DEFAULT '',
                    age INTEGER,
                    contact TEXT NOT NULL DEFAULT '',
                    visit_date TEXT NOT NULL DEFAULT '',
                    department TEXT NOT NULL DEFAULT '',
                    primary_complaint TEXT NOT NULL DEFAULT '',
                    ultrasound_description TEXT NOT NULL DEFAULT '',
                    birads_category TEXT NOT NULL DEFAULT '',
                    pathology_status TEXT NOT NULL DEFAULT '',
                    risk_level TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_patient_cases_code ON patient_cases(patient_code)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_patient_cases_updated_at ON patient_cases(updated_at)"
            )

    @staticmethod
    def _row_to_case(row: sqlite3.Row) -> PatientCase:
        """Convert a SQLite row to the public case model."""
        return PatientCase(**dict(row))

    def create_case(self, payload: PatientCaseCreate) -> PatientCase:
        """Insert a new patient case."""
        values = _model_to_dict(payload)
        now = datetime.now(UTC).isoformat()
        values["created_at"] = now
        values["updated_at"] = now
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        with self._connect() as connection:
            cursor = connection.execute(
                f"INSERT INTO patient_cases ({columns}) VALUES ({placeholders})",
                list(values.values()),
            )
            row = connection.execute(
                "SELECT * FROM patient_cases WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return self._row_to_case(row)

    def seed_sample_cases(self) -> list[PatientCase]:
        """Insert demo patient cases if their patient codes do not already exist."""
        created_or_existing: list[PatientCase] = []
        with self._connect() as connection:
            for sample in SAMPLE_PATIENT_CASES:
                existing = connection.execute(
                    "SELECT * FROM patient_cases WHERE patient_code = ?",
                    (sample.patient_code,),
                ).fetchone()
                if existing is not None:
                    created_or_existing.append(self._row_to_case(existing))
                    continue

                values = _model_to_dict(sample)
                now = datetime.now(UTC).isoformat()
                values["created_at"] = now
                values["updated_at"] = now
                columns = ", ".join(values.keys())
                placeholders = ", ".join("?" for _ in values)
                cursor = connection.execute(
                    f"INSERT INTO patient_cases ({columns}) VALUES ({placeholders})",
                    list(values.values()),
                )
                row = connection.execute(
                    "SELECT * FROM patient_cases WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
                created_or_existing.append(self._row_to_case(row))
        return created_or_existing

    def list_cases(self, query: str | None = None) -> list[PatientCase]:
        """List patient cases, optionally filtering by patient code/name/diagnosis text."""
        parameters: list[Any] = []
        where_clause = ""
        if query:
            pattern = f"%{query.strip()}%"
            where_clause = """
                WHERE patient_code LIKE ?
                   OR name LIKE ?
                   OR ultrasound_description LIKE ?
                   OR birads_category LIKE ?
                   OR risk_level LIKE ?
            """
            parameters = [pattern, pattern, pattern, pattern, pattern]
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM patient_cases
                {where_clause}
                ORDER BY updated_at DESC, id DESC
                """,
                parameters,
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def get_case(self, case_id: int) -> PatientCase | None:
        """Return a patient case by ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM patient_cases WHERE id = ?",
                (case_id,),
            ).fetchone()
        return self._row_to_case(row) if row is not None else None
