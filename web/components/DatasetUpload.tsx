"use client";

import { useId, useState } from "react";

import { ApiError, uploadDataset } from "@/lib/api";
import { DATASET_ID_PATTERN, type DatasetSummary } from "@/lib/types";

interface DatasetUploadProps {
  onUploaded: (dataset: DatasetSummary) => void;
}

export function DatasetUpload({ onUploaded }: DatasetUploadProps) {
  const formId = useId();
  const [datasetId, setDatasetId] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const datasetIdValid = datasetId.length > 0 && DATASET_ID_PATTERN.test(datasetId);
  const canSubmit = datasetIdValid && description.trim().length > 0 && file !== null;

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit || !file) return;

    setStatus("uploading");
    setError(null);
    try {
      const result = await uploadDataset(file, datasetId, description.trim());
      onUploaded({ dataset_id: result.dataset_id, description: result.description });
      setDatasetId("");
      setDescription("");
      setFile(null);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setError(err instanceof ApiError ? err.message : "Upload failed. Please try again.");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex min-w-0 flex-1 flex-col gap-2.5">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
        Upload a CSV dataset
      </span>

      <input
        type="text"
        placeholder="dataset id (letters, numbers, -, _)"
        value={datasetId}
        onChange={(event) => setDatasetId(event.target.value)}
        className="w-full min-w-0 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-800 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
        aria-label="Dataset id"
      />
      {datasetId.length > 0 && !datasetIdValid && (
        <p className="text-xs text-red-600 dark:text-red-400">
          Only letters, numbers, hyphens, and underscores are allowed.
        </p>
      )}

      <input
        type="text"
        placeholder="short description"
        value={description}
        onChange={(event) => setDescription(event.target.value)}
        className="w-full min-w-0 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-800 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
        aria-label="Dataset description"
      />

      <input
        id={`${formId}-file`}
        type="file"
        accept=".csv,text/csv"
        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        className="text-sm text-zinc-600 file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-100 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-zinc-700 dark:text-zinc-400 dark:file:bg-zinc-800 dark:file:text-zinc-200"
      />

      <button
        type="submit"
        disabled={!canSubmit || status === "uploading"}
        className="rounded-lg bg-zinc-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
      >
        {status === "uploading" ? "Uploading…" : "Upload dataset"}
      </button>

      {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
    </form>
  );
}
