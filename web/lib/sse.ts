import type { StreamEvent } from "./types";

/**
 * Parse a `text/event-stream` body into typed events.
 *
 * The browser's native `EventSource` can't be used here -- it's GET-only and can't send the
 * JSON body `POST /chat/stream` needs -- so this reads the fetch `Response.body`
 * `ReadableStream` directly. Frames (`data: <json>\n\n`) can arrive split across two chunks, so
 * partial text is buffered across reads and only complete frames are parsed and yielded.
 */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? ""; // last element is either "" or an incomplete trailing frame

      for (const frame of frames) {
        const event = parseFrame(frame);
        if (event) yield event;
      }
    }
    // A final frame with no trailing "\n\n" (e.g. the stream closed right after it) is still
    // real data -- don't drop it.
    const trailing = parseFrame(buffer);
    if (trailing) yield trailing;
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): StreamEvent | null {
  const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
  if (!dataLine) return null;
  try {
    return JSON.parse(dataLine.slice("data: ".length)) as StreamEvent;
  } catch {
    return null; // malformed frame -- skip rather than crash the whole stream
  }
}
