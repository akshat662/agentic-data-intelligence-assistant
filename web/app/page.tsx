"use client";

import { useState } from "react";

import { ChatWindow } from "@/components/ChatWindow";
import { DatasetSelector } from "@/components/DatasetSelector";
import { DatasetUpload } from "@/components/DatasetUpload";
import type { DatasetSummary } from "@/lib/types";

// The one dataset this repo ships and registers by default (see `data/registry.json`) --
// real, not mock data. There is no `GET /datasets` endpoint yet, so this is the seed for a
// session-local list that grows as datasets are uploaded (see `DatasetSelector`'s docstring).
const DEFAULT_DATASETS: DatasetSummary[] = [
  {
    dataset_id: "superstore",
    description:
      "US retail order-level sales data (Sample - Superstore): 9,994 orders across " +
      "Furniture, Office Supplies, and Technology.",
  },
];

export default function Home() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>(DEFAULT_DATASETS);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    DEFAULT_DATASETS[0]?.dataset_id ?? null,
  );

  function handleUploaded(dataset: DatasetSummary) {
    setDatasets((prev) => [
      ...prev.filter((existing) => existing.dataset_id !== dataset.dataset_id),
      dataset,
    ]);
    setSelectedDatasetId(dataset.dataset_id);
  }

  return (
    <div className="mx-auto flex h-screen w-full max-w-4xl flex-col gap-5 p-6">
      <header>
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">
          ADIA — Agentic Data Intelligence Assistant
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Ask an analytical question and watch a grounded, evidence-cited investigation run live.
        </p>
      </header>

      <div className="flex flex-col gap-4 rounded-2xl border border-zinc-200 bg-white p-4 sm:flex-row sm:items-start sm:justify-between dark:border-zinc-800 dark:bg-zinc-950">
        <DatasetSelector
          datasets={datasets}
          selectedDatasetId={selectedDatasetId}
          onSelect={setSelectedDatasetId}
        />
        <DatasetUpload onUploaded={handleUploaded} />
      </div>

      <ChatWindow datasetId={selectedDatasetId} />
    </div>
  );
}
