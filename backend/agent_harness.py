"""DeepSeek-backed breast tumor interpretation agent harness."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from backend.knowledge import KnowledgeEntry, build_knowledge_context, search_knowledge
from backend.patients import PatientCase


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_AGENT_PROMPT_PATH = Path(__file__).with_name("agent.md")
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
ID_CARD_PATTERN = re.compile(r"(?<!\w)\d{17}[\dXx](?!\w)")
CASE_CODE_PATTERN = re.compile(r"(?<!\w)(?:BUCAD|DEMO)-[A-Z0-9-]+(?!\w)")


class AgentInterpretationRequest(BaseModel):
    """Request accepted by the medical interpretation agent."""

    question: str = Field("", max_length=2000)
    patient_case: PatientCase | None = None
    diagnosis: dict[str, Any] | None = None


class AgentInterpretationResponse(BaseModel):
    """Agent response returned to the Vue frontend."""

    answer: str
    model: str
    provider: str
    fallback_used: bool
    retrieved_knowledge: list[KnowledgeEntry]
    warnings: list[str] = Field(default_factory=list)


def _read_agent_prompt(path: Path = DEFAULT_AGENT_PROMPT_PATH) -> str:
    """Read the agent instruction file."""
    return path.read_text(encoding="utf-8")


def _deepseek_base_url() -> str:
    """Return configured DeepSeek base URL without a trailing slash."""
    return os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")


def _deepseek_model() -> str:
    """Return configured DeepSeek model."""
    return os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def _diagnosis_summary(diagnosis: dict[str, Any] | None) -> str:
    """Format frontend diagnosis payload for prompt context."""
    if not diagnosis:
        return "暂无本次图像 AI 诊断结果。"
    result = diagnosis.get("result") if isinstance(diagnosis, dict) else None
    if not isinstance(result, dict):
        return "暂无本次图像 AI 诊断结果。"
    malignant = result.get("malignant_probability")
    benign = result.get("benign_probability")
    label = result.get("final_label_text") or result.get("final_label") or "未给出"
    confidence = result.get("confidence_band_text") or result.get("confidence_band") or "未给出"
    model_name = result.get("model_display_name") or result.get("model_version") or "未给出"
    lines = [
        f"AI 判定：{label}",
        f"置信度：{confidence}",
        f"恶性概率：{float(malignant) * 100:.1f}%" if isinstance(malignant, (int, float)) else "恶性概率：未给出",
        f"良性概率：{float(benign) * 100:.1f}%" if isinstance(benign, (int, float)) else "良性概率：未给出",
        f"模型：{model_name}",
    ]
    return "\n".join(lines)


def _redact_text(value: str | None, patient_name: str = "") -> str:
    """Remove direct identifiers before text is sent to the remote agent."""
    if not value:
        return "未填"
    redacted = str(value)
    if patient_name:
        redacted = redacted.replace(patient_name, "该患者")
    redacted = PHONE_PATTERN.sub("[已脱敏联系方式]", redacted)
    redacted = EMAIL_PATTERN.sub("[已脱敏邮箱]", redacted)
    redacted = ID_CARD_PATTERN.sub("[已脱敏证件号]", redacted)
    redacted = CASE_CODE_PATTERN.sub("[已脱敏病例编号]", redacted)
    return redacted.strip() or "未填"


def _age_band(age: int | None) -> str:
    """Return a clinically useful age range without exposing exact age."""
    if age is None:
        return "未填"
    if age < 30:
        return "30岁以下"
    if age >= 80:
        return "80岁及以上"
    decade = age // 10 * 10
    return f"{decade}-{decade + 9}岁"


def _patient_case_summary(patient_case: PatientCase | None) -> str:
    """Format a de-identified patient case for prompt context."""
    if patient_case is None:
        return "未选择病例。"
    patient_name = patient_case.name
    return "\n".join(
        [
            "病例称呼：本例（已脱敏）",
            f"性别/年龄段：{patient_case.sex} / {_age_band(patient_case.age)}",
            f"科室：{_redact_text(patient_case.department, patient_name)}",
            f"BI-RADS：{_redact_text(patient_case.birads_category, patient_name)}",
            f"病理状态：{_redact_text(patient_case.pathology_status, patient_name)}",
            f"风险等级：{_redact_text(patient_case.risk_level, patient_name)}",
            f"主诉：{_redact_text(patient_case.primary_complaint, patient_name)}",
            f"超声描述：{_redact_text(patient_case.ultrasound_description, patient_name)}",
            f"备注：{_redact_text(patient_case.notes, patient_name)}",
        ]
    )


def _build_retrieval_query(request: AgentInterpretationRequest) -> str:
    """Build a compact retrieval query from case, diagnosis, and user question."""
    case = request.patient_case
    patient_name = case.name if case is not None else ""
    fields = [_redact_text(request.question, patient_name)]
    if case is not None:
        fields.extend(
            [
                _redact_text(case.primary_complaint, patient_name),
                _redact_text(case.ultrasound_description, patient_name),
                _redact_text(case.birads_category, patient_name),
                _redact_text(case.pathology_status, patient_name),
                _redact_text(case.risk_level, patient_name),
                _redact_text(case.notes, patient_name),
            ]
        )
    if request.diagnosis:
        fields.append(_redact_text(json.dumps(request.diagnosis, ensure_ascii=False), patient_name))
    return " ".join(str(field) for field in fields if field)


def _build_user_prompt(
    request: AgentInterpretationRequest,
    retrieved_knowledge: list[KnowledgeEntry],
) -> str:
    """Build the user message sent to the LLM."""
    patient_name = request.patient_case.name if request.patient_case is not None else ""
    question = _redact_text(request.question.strip(), patient_name)
    if question == "未填":
        question = "请基于当前病例和 AI 结果生成辅助诊断建议。"
    return (
        "请根据以下资料生成辅助诊断建议。\n\n"
        f"医生问题：\n{question}\n\n"
        f"病例信息：\n{_patient_case_summary(request.patient_case)}\n\n"
        f"图像 AI 结果：\n{_diagnosis_summary(request.diagnosis)}\n\n"
        "内部知识库上下文（仅用于推理和事实核对，回答中不要单列依据章节）：\n"
        f"{build_knowledge_context(retrieved_knowledge)}"
    )


def _fallback_answer(
    request: AgentInterpretationRequest,
    retrieved_knowledge: list[KnowledgeEntry],
    reason: str,
) -> str:
    """Return a deterministic local answer when the remote LLM is unavailable."""
    del retrieved_knowledge
    case = request.patient_case
    diagnosis = request.diagnosis or {}
    result = diagnosis.get("result") if isinstance(diagnosis, dict) else None
    label = result.get("final_label_text") if isinstance(result, dict) else None
    malignant = result.get("malignant_probability") if isinstance(result, dict) else None
    malignant_text = (
        f"恶性概率约 {float(malignant) * 100:.1f}%"
        if isinstance(malignant, (int, float))
        else "暂未提供恶性概率"
    )
    birads = case.birads_category if case else ""
    risk = case.risk_level if case else ""
    high_risk_hint = "BI-RADS 4/5 或高风险病例建议优先组织学确认。" if (
        "4" in birads or "5" in birads or "高" in risk
    ) else "建议结合 BI-RADS 和影像变化决定随访或进一步检查。"
    return "\n".join(
        [
            "## 🧾 1. 结论摘要",
            f"- 🎯 **初步判断**：AI 图像结果为 {label or '未完成图像诊断'}，{malignant_text}；需结合 BI-RADS、超声征象和病理结果复核。",
            f"- 🧭 **优先事项**：{high_risk_hint}",
            f"- 📌 **当前状态**：DeepSeek 调用未完成，已返回本地结构化草稿；原因：{reason}",
            "",
            "## 🖼️ 2. 影像与 AI 结果解读",
            f"- 🔎 **超声与分级**：本例 BI-RADS 为 {birads or '未填写'}，风险等级为 {risk or '未评估'}；若存在边界欠清、形态不规则、非平行生长、后方声影或可疑淋巴结，应提高复核优先级。",
            "- 🤖 **AI 模型输出**：模型概率只能作为影像复核线索，不能等同病理诊断；当模型结果与 BI-RADS 或医生描述不一致时，应优先核对图像质量、病灶框定和检查报告。",
            "- 🧩 **可解释性**：若热力图与病灶区域及可疑超声征象一致，可作为关注区域提示；若关注区域偏离病灶，应降低解释权重。",
            "",
            "## 🔍 3. 需要补充或复核的信息",
            "- 🧪 **病理与免疫组化**：建议补充组织学类型、分级、ER、PR、HER2、Ki-67、脉管侵犯和取材代表性。",
            "- 🧬 **分期与转移评估**：若倾向恶性或 BI-RADS 4/5，应补充腋窝淋巴结评估、TNM 分期和必要的全身评估。",
            "- 📝 **影像资料**：建议补充病灶大小、位置、边界、形态、纵横比、后方回声、血流和既往影像变化。",
            "",
            "## 🩺 4. 辅助建议",
            f"- ✅ **诊断路径**：{high_risk_hint}",
            "- 🏥 **协作复核**：影像高度可疑但病理良性时，应评估影像病理一致性和取材代表性，必要时组织乳腺外科、超声科、病理科和肿瘤内科讨论。",
            "- 📊 **随访管理**：低风险或资料不足病例应结合 BI-RADS 分类、既往影像和临床体征制定复查计划。",
            "- 💊 **治疗提示**：药物治疗方向需由专科医生基于病理分型、分期、受体状态和指南综合评估。",
            "",
            "## ⚠️ 5. 风险提示",
            "- 🚨 **高风险信号**：BI-RADS 4/5、恶性概率高、腋窝淋巴结可疑、影像病理不一致时，需要提高处理优先级。",
            "- 🧯 **不确定性来源**：图像质量、模型置信度、病例资料缺失和取材偏倚都可能影响判断。",
            "- 👩‍⚕️ **最终决策**：本输出仅用于辅助分析和科研演示，不能替代医生诊断、病理报告或正式诊疗建议。",
        ]
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _call_deepseek(messages: list[dict[str, str]]) -> str:
    """Call DeepSeek's OpenAI-compatible chat completion API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")

    payload = {
        "model": _deepseek_model(),
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1800,
        "thinking": {"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
    }
    request = urllib.request.Request(
        f"{_deepseek_base_url()}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API 请求失败：HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek API 连接失败：{exc.reason}") from exc

    choices = body.get("choices")
    if not choices:
        raise RuntimeError("DeepSeek API 未返回 choices。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError("DeepSeek API 未返回文本内容。")
    return str(content)


def _deepseek_delta_from_chunk(line: str) -> str:
    """Extract text delta from one DeepSeek/OpenAI-compatible stream line."""
    if not line.startswith("data:"):
        return ""
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return ""
    body = json.loads(payload)
    choices = body.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return str(content) if content else ""


def _stream_deepseek(messages: list[dict[str, str]]) -> Iterator[str]:
    """Stream text chunks from DeepSeek's OpenAI-compatible chat completion API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")

    payload = {
        "model": _deepseek_model(),
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1800,
        "stream": True,
        "thinking": {"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
    }
    request = urllib.request.Request(
        f"{_deepseek_base_url()}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line == "data: [DONE]":
                    break
                delta = _deepseek_delta_from_chunk(line)
                if delta:
                    yield delta
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API 请求失败：HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek API 连接失败：{exc.reason}") from exc


class BreastTumorAgentHarness:
    """Retrieve local knowledge and ask DeepSeek for a structured interpretation."""

    def __init__(self, agent_prompt_path: Path = DEFAULT_AGENT_PROMPT_PATH) -> None:
        """Store the prompt path used by each interpretation request."""
        self.agent_prompt_path = agent_prompt_path

    def interpret(self, request: AgentInterpretationRequest) -> AgentInterpretationResponse:
        """Generate an agent answer with local knowledge retrieval."""
        retrieval_query = _build_retrieval_query(request)
        retrieved_knowledge = search_knowledge(retrieval_query, limit=5)
        warnings: list[str] = []
        try:
            system_prompt = _read_agent_prompt(self.agent_prompt_path)
            answer = _call_deepseek(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_user_prompt(request, retrieved_knowledge)},
                ]
            )
            return AgentInterpretationResponse(
                answer=answer,
                model=_deepseek_model(),
                provider="deepseek",
                fallback_used=False,
                retrieved_knowledge=retrieved_knowledge,
                warnings=warnings,
            )
        except Exception as exc:
            reason = str(exc)
            warnings.append(reason)
            return AgentInterpretationResponse(
                answer=_fallback_answer(request, retrieved_knowledge, reason),
                model=_deepseek_model(),
                provider="local-fallback",
                fallback_used=True,
                retrieved_knowledge=retrieved_knowledge,
                warnings=warnings,
            )

    def interpret_stream(self, request: AgentInterpretationRequest) -> Iterator[str]:
        """Generate an SSE stream with retrieval metadata and text deltas."""
        retrieval_query = _build_retrieval_query(request)
        retrieved_knowledge = search_knowledge(retrieval_query, limit=5)
        knowledge_payload = [
            entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
            for entry in retrieved_knowledge
        ]
        yield _sse_event(
            "metadata",
            {
                "model": _deepseek_model(),
                "provider": "deepseek",
                "fallback_used": False,
                "retrieved_knowledge": knowledge_payload,
                "warnings": [],
            },
        )
        try:
            system_prompt = _read_agent_prompt(self.agent_prompt_path)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_prompt(request, retrieved_knowledge)},
            ]
            answer_parts: list[str] = []
            for delta in _stream_deepseek(messages):
                answer_parts.append(delta)
                yield _sse_event("delta", {"text": delta})
            yield _sse_event(
                "complete",
                {
                    "answer": "".join(answer_parts),
                    "model": _deepseek_model(),
                    "provider": "deepseek",
                    "fallback_used": False,
                    "retrieved_knowledge": knowledge_payload,
                    "warnings": [],
                },
            )
        except Exception as exc:
            reason = str(exc)
            answer = _fallback_answer(request, retrieved_knowledge, reason)
            yield _sse_event(
                "fallback",
                {
                    "answer": answer,
                    "model": _deepseek_model(),
                    "provider": "local-fallback",
                    "fallback_used": True,
                    "retrieved_knowledge": knowledge_payload,
                    "warnings": [reason],
                },
            )
        yield _sse_event("done", {})
