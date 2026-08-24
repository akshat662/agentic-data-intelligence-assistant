import { parseSSE } from "./sse";
import type { ChatRequest, DatasetUploadResponse, StreamEvent } from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Thrown for any non-2xx response from the API. Carries the server's `{detail}` when present. */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? response.statusText;
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

/** Upload a CSV and register it as a new dataset. Real call only -- no mock data. */
export async function uploadDataset(
  file: File,
  datasetId: string,
  description: string,
): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("dataset_id", datasetId);
  form.append("description", description);
  form.append("file", file);

  const response = await fetch(`${API_BASE_URL}/datasets`, { method: "POST", body: form });
  if (!response.ok) {
    throw new ApiError(await errorDetail(response), response.status);
  }
  return (await response.json()) as DatasetUploadResponse;
}

/**
 * Run a question through the ADIA graph, streaming progress as it goes.
 *
 * A non-2xx response (e.g. a 422 from a malformed request) is a plain JSON error, not an SSE
 * body, and is thrown as an `ApiError` rather than handed to the SSE parser. A run that fails
 * *after* the stream has started (an unreachable LLM, an unhandled graph error) is not an HTTP
 * error at all -- the backend always yields a `StreamErrorEvent` frame instead, which callers
 * see as a normal event with `type: "error"`.
 */
export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new ApiError(await errorDetail(response), response.status);
  }
  if (!response.body) {
    throw new ApiError("Response had no body to stream.", response.status);
  }

  yield* parseSSE(response.body);
}
