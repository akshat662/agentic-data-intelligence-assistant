"use client";

import { useCallback, useReducer, useRef } from "react";

import { ApiError, streamChat } from "@/lib/api";
import type {
  RenderedEvidence,
  StreamEvidenceEvent,
  StreamFinalEvent,
  StreamPhaseEvent,
} from "@/lib/types";

export interface Turn {
  id: string;
  question: string;
  phases: StreamPhaseEvent[];
  evidence: RenderedEvidence[];
  final: StreamFinalEvent | null;
  error: string | null;
}

interface ChatState {
  turns: Turn[];
  isStreaming: boolean;
}

type Action =
  | { kind: "start"; id: string; question: string }
  | { kind: "phase"; id: string; event: StreamPhaseEvent }
  | { kind: "evidence"; id: string; event: StreamEvidenceEvent }
  | { kind: "final"; id: string; event: StreamFinalEvent }
  | { kind: "error"; id: string; detail: string };

const initialState: ChatState = { turns: [], isStreaming: false };

function updateTurn(state: ChatState, id: string, update: (turn: Turn) => Turn): ChatState {
  return {
    ...state,
    turns: state.turns.map((turn) => (turn.id === id ? update(turn) : turn)),
  };
}

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.kind) {
    case "start":
      return {
        isStreaming: true,
        turns: [
          ...state.turns,
          {
            id: action.id,
            question: action.question,
            phases: [],
            evidence: [],
            final: null,
            error: null,
          },
        ],
      };
    case "phase":
      return updateTurn(state, action.id, (turn) => ({
        ...turn,
        phases: [...turn.phases, action.event],
      }));
    case "evidence":
      return updateTurn(state, action.id, (turn) => ({
        ...turn,
        evidence: [...turn.evidence, action.event.evidence],
      }));
    case "final":
      return {
        ...updateTurn(state, action.id, (turn) => ({ ...turn, final: action.event })),
        isStreaming: false,
      };
    case "error":
      return {
        ...updateTurn(state, action.id, (turn) => ({ ...turn, error: action.detail })),
        isStreaming: false,
      };
    default:
      return state;
  }
}

/**
 * Drives one chat "session" against `POST /chat/stream` -- a client-side-only illusion of a
 * conversation: every `ask()` call is an independent, stateless graph run (ADIA never feeds
 * prior turns back into the LLM), the hook just keeps the running transcript for display.
 */
export function useChat(datasetId: string | null) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const abortRef = useRef<AbortController | null>(null);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!datasetId || !trimmed || state.isStreaming) return;

      const id = crypto.randomUUID();
      dispatch({ kind: "start", id, question: trimmed });

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const event of streamChat(
          { dataset_id: datasetId, question: trimmed },
          controller.signal,
        )) {
          if (event.type === "phase") dispatch({ kind: "phase", id, event });
          else if (event.type === "evidence") dispatch({ kind: "evidence", id, event });
          else if (event.type === "final") dispatch({ kind: "final", id, event });
          else dispatch({ kind: "error", id, detail: event.detail });
        }
      } catch (err) {
        const detail =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Something went wrong.";
        dispatch({ kind: "error", id, detail });
      }
    },
    [datasetId, state.isStreaming],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { turns: state.turns, isStreaming: state.isStreaming, ask, cancel };
}
