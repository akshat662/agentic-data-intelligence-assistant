# Data

## Superstore Dataset

### Description

`superstore.csv` is the widely used **"Sample - Superstore"** dataset: 9,994 US retail order
line items spanning three product categories (Furniture, Office Supplies, Technology),
four regions, and roughly three years of orders. Each row is one order line — a specific
product on a specific order — with its sales revenue, quantity, discount, and profit.

It is not a synthetic or ADIA-authored dataset. It's one of the most widely used teaching
datasets in the BI/analytics space, originally distributed as Tableau's built-in sample
workbook and mirrored publicly on GitHub and Kaggle in thousands of tutorials and student
projects for exactly this reason: a real, analyzable retail dataset without licensing
friction.

### Source

Downloaded from the public GitHub mirror
[`leonism/sample-superstore`](https://github.com/leonism/sample-superstore/blob/master/data/superstore.csv).
That mirror's CSV file had two unrelated tables (a "Returns" sheet and a "People" sheet
from the original multi-sheet Excel workbook) appended directly below the Orders table with
no section break — a copy-paste artifact from that specific upload, not a property of the
dataset itself. `data/superstore.csv` here is the Orders table only: the first 9,994 data
rows, verified against the dataset's well-documented canonical shape (9,994 rows, 21
columns, 3 categories, 4 regions, 3 segments, 4 ship modes). No row or column of the Orders
table was modified.

### Why This Dataset Fits This Project

- **Genuinely supports all five answerable question categories** the tool layer
  (`adia/tools/`) already implements: aggregation by category/region (`run_sql`), dataset
  shape and summary (`profile_dataset`), group comparisons and correlations
  (`compare_groups`, `compute_correlation`), root-cause–style questions (why did profit in
  one region differ from another), and prediction (`train_model` — `Profit` and `Sales` are
  natural regression targets from `Quantity`/`Discount`; `Region`/`Segment`/`Ship Mode` are
  natural classification targets).
- **Has real structure worth discovering, not just numbers to look up.** Discount visibly
  interacts with Profit (a large discount can make an individual line item unprofitable),
  which is exactly the kind of relationship a correlation/group-comparison tool should
  surface — and exactly the kind of relationship a static validator must stop the system
  from overclaiming as *causal*.
- **Small enough to be a good final-year-scope demo.** ~2 MB, loads and profiles in well
  under a second, no distributed processing or sampling story needed.
- **Genuinely lacks information for some questions**, which is what makes the refusal /
  unanswerable question category meaningful rather than contrived: there is no employee or
  sales-rep column, no customer contact information, no return-reason data (deliberately
  excluded during cleaning, see above), and obviously no future or competitor data. A system
  asked those questions has to decline, not guess.

### Schema Overview

21 columns, one row per order line item. As loaded by the existing generic CSV loader
(`adia.data.loader.load_dataset`) with no dataset-specific parsing — see the note on `Order
Date`/`Ship Date` below.

| Column | Type (as loaded) | Notes |
|---|---|---|
| `Row ID` | numeric (int) | Unique row identifier, 1–9994. |
| `Order ID` | categorical | Shared by every line item in the same order (5,009 distinct orders). |
| `Order Date`, `Ship Date` | categorical (string) | Stored as `M/D/YYYY` text. The generic loader does not parse dates, so these profile as high-cardinality categorical columns rather than `datetime` — a real, observed property of the current dataset-agnostic pipeline, not specific to this dataset. Date-based SQL queries can still `CAST`/`strptime` these at query time. |
| `Ship Mode` | categorical | 4 values: Standard/Second/First Class, Same Day. |
| `Customer ID`, `Customer Name` | categorical | 793 distinct customers. |
| `Segment` | categorical | 3 values: Consumer, Corporate, Home Office. |
| `Country` | categorical | Single value ("United States") — present in the schema, but carries no analytical signal. |
| `City`, `State` | categorical | 531 / 49 distinct values. |
| `Postal Code` | numeric | 11 missing values (0.11%) — the only column with any missingness. |
| `Region` | categorical | 4 values: Central, East, South, West. |
| `Product ID`, `Product Name` | categorical | 1,862 / 1,850 distinct products. |
| `Category` | categorical | 3 values: Furniture, Office Supplies, Technology. |
| `Sub-Category` | categorical | 17 values nested under `Category`. |
| `Sales` | numeric (float) | Line-item revenue, $0.44–$22,638.48. |
| `Quantity` | numeric (int) | Units ordered, 1–14. |
| `Discount` | numeric (float) | 0.0–0.8 (0%–80%). |
| `Profit` | numeric (float) | Can be negative — $-6,599.98 to $8,399.98. |

The generated catalog (column-level dtype, semantic type, null rate, cardinality, numeric
range) is checked in at [`data/catalog/superstore.json`](catalog/superstore.json), produced
by `adia.data.catalog.build_catalog` — the same profiling code path used everywhere else in
this project, not a dataset-specific script.

### Registration

Registered under dataset ID `superstore` in [`data/registry.json`](registry.json) via the
existing `DatasetConfig`/`adia.data.registry` mechanism (Phase 2B) — no dataset-specific code
was added anywhere in `adia/tools/` or `adia/data/` to support this dataset.
