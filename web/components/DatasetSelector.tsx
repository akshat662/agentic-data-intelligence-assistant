import type { DatasetSummary } from "@/lib/types";

interface DatasetSelectorProps {
  datasets: DatasetSummary[];
  selectedDatasetId: string | null;
  onSelect: (datasetId: string) => void;
}

/**
 * Picks among datasets known to this session. `datasets` is passed in as a prop rather than
 * fetched here on purpose: there is no `GET /datasets` endpoint yet, so `app/page.tsx` currently
 * seeds this list itself (the shipped `superstore` dataset + anything uploaded this session).
 * When that endpoint exists, only `page.tsx`'s data source changes -- this component's contract
 * (a list of `{dataset_id, description}` in, a selection callback out) already matches what a
 * real fetch would return.
 */
export function DatasetSelector({ datasets, selectedDatasetId, onSelect }: DatasetSelectorProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor="dataset-select" className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
        Dataset
      </label>
      <select
        id="dataset-select"
        value={selectedDatasetId ?? ""}
        onChange={(event) => onSelect(event.target.value)}
        className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-800 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
      >
        {datasets.length === 0 && <option value="">No datasets yet</option>}
        {datasets.map((dataset) => (
          <option key={dataset.dataset_id} value={dataset.dataset_id}>
            {dataset.dataset_id} — {dataset.description}
          </option>
        ))}
      </select>
    </div>
  );
}
