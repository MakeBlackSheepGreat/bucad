import type {
  AgentInterpretationRequest,
  AgentInterpretationResponse,
  DiagnosisPayload,
  KnowledgeEntry,
  PatientCase,
  PatientCaseInput,
  RuntimeInfo,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.json().catch(() => null);
  if (typeof body?.detail === "string") {
    return body.detail;
  }
  if (typeof body?.message === "string") {
    return body.message;
  }
  return `请求失败：${response.status}`;
}

export async function getRuntimeInfo(): Promise<RuntimeInfo> {
  const response = await fetch(`${API_BASE_URL}/api/runtime`);
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<RuntimeInfo>;
}

export async function diagnoseImage(params: {
  file: File;
  threshold: number;
  needSegmentation: boolean;
  needExplanation: boolean;
}): Promise<DiagnosisPayload> {
  const form = new FormData();
  form.append("file", params.file);
  form.append("threshold", String(params.threshold));
  form.append("need_segmentation", String(params.needSegmentation));
  form.append("need_explanation", String(params.needExplanation));

  const response = await fetch(`${API_BASE_URL}/api/diagnose`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<DiagnosisPayload>;
}

export async function listPatientCases(query = ""): Promise<PatientCase[]> {
  const search = query.trim() ? `?q=${encodeURIComponent(query.trim())}` : "";
  const response = await fetch(`${API_BASE_URL}/api/patients${search}`);
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<PatientCase[]>;
}

export async function createPatientCase(payload: PatientCaseInput): Promise<PatientCase> {
  const response = await fetch(`${API_BASE_URL}/api/patients`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<PatientCase>;
}

export async function seedSamplePatientCases(): Promise<PatientCase[]> {
  const response = await fetch(`${API_BASE_URL}/api/patients/seed-samples`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<PatientCase[]>;
}

export async function listKnowledgeEntries(params: {
  query?: string;
  category?: string;
} = {}): Promise<KnowledgeEntry[]> {
  const search = new URLSearchParams();
  if (params.query?.trim()) {
    search.set("q", params.query.trim());
  }
  if (params.category?.trim()) {
    search.set("category", params.category.trim());
  }
  const suffix = search.toString() ? `?${search.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/api/knowledge${suffix}`);
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<KnowledgeEntry[]>;
}

export async function interpretWithAgent(
  payload: AgentInterpretationRequest,
): Promise<AgentInterpretationResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agent/interpret`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return response.json() as Promise<AgentInterpretationResponse>;
}

export async function streamInterpretWithAgent(
  payload: AgentInterpretationRequest,
  handlers: {
    onMetadata?: (metadata: Omit<AgentInterpretationResponse, "answer">) => void;
    onDelta?: (text: string) => void;
    onComplete?: (response: AgentInterpretationResponse) => void;
    onFallback?: (response: AgentInterpretationResponse) => void;
  },
): Promise<AgentInterpretationResponse | null> {
  const response = await fetch(`${API_BASE_URL}/api/agent/interpret/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  if (!response.body) {
    throw new ApiError(response.status, "智能体流式响应为空。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalResponse: AgentInterpretationResponse | null = null;

  function consumeEvent(rawEvent: string): void {
    const lines = rawEvent.split(/\r?\n/);
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) {
      return;
    }
    const data = JSON.parse(dataLines.join("\n"));
    if (eventName === "metadata") {
      handlers.onMetadata?.(data as Omit<AgentInterpretationResponse, "answer">);
    } else if (eventName === "delta") {
      handlers.onDelta?.(String(data.text ?? ""));
    } else if (eventName === "complete") {
      finalResponse = data as AgentInterpretationResponse;
      handlers.onComplete?.(finalResponse);
    } else if (eventName === "fallback") {
      finalResponse = data as AgentInterpretationResponse;
      handlers.onFallback?.(finalResponse);
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    let separatorIndex = buffer.search(/\r?\n\r?\n/);
    while (separatorIndex >= 0) {
      const rawEvent = buffer.slice(0, separatorIndex);
      const match = buffer.slice(separatorIndex).match(/^\r?\n\r?\n/);
      buffer = buffer.slice(separatorIndex + (match?.[0].length ?? 2));
      consumeEvent(rawEvent);
      separatorIndex = buffer.search(/\r?\n\r?\n/);
    }
    if (done) {
      break;
    }
  }
  if (buffer.trim()) {
    consumeEvent(buffer);
  }
  return finalResponse;
}
