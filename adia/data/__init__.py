"""Data layer: dataset loading and catalog (schema/profile) generation.

No dataset-specific cleaning or ingestion pipelines live here yet — only the generic,
dataset-agnostic loading and profiling primitives that dataset-specific ingest scripts will
build on.
"""

from adia.data.catalog import build_catalog, load_catalog, profile_column, save_catalog
from adia.data.loader import load_dataset

__all__ = [
    "build_catalog",
    "load_catalog",
    "load_dataset",
    "profile_column",
    "save_catalog",
]
