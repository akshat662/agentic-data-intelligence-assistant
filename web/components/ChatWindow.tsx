"use client";

import { useState } from "react";

import { useChat } from "@/hooks/useChat";

import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  datasetId: string | null;
}

export function ChatWindow({ datasetId }: ChatWindowProps) {
  const { turns, isStreaming, ask } = useChat(datasetId);
  const [question, setQuestion] = useState("");

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || isStreaming || !datasetId) return;
    void ask(question);
    setQuestion("");
  }

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-hidden">
      <div className="flex flex-1 flex-col gap-6 overflow-y-auto rounded-2xl border border-zinc-200 bg-zinc-50/50 p-4 dark:border-zinc-800 dark:bg-zinc-900/30">
        {turns.length === 0 && (
          <p className="m-auto max-w-sm text-center text-sm text-zinc-400 dark:text-zinc-500">
            {datasetId
              ? "Ask an analytical question about the selected dataset to start an investigation."
              : "Select or upload a dataset to get started."}
          </p>
        )}
        {turns.map((turn, index) => (
          <MessageBubble key={turn.id} turn={turn} isActive={index === turns.length - 1} />
        ))}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={
            datasetId ? "Ask a question about the dataset…" : "Select a dataset first…"
          }
          disabled={!datasetId || isStreaming}
          className="flex-1 rounded-full border border-zinc-200 bg-white px-4 py-2.5 text-sm text-zinc-800 disabled:opacity-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
        />
        <button
          type="submit"
          disabled={!datasetId || isStreaming || !question.trim()}
          className="rounded-full bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {isStreaming ? "Investigating…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
