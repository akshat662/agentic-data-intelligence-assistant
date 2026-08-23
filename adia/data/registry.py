"""Dataset registry: where future datasets are declared, before any file needs to exist.

The registry is purely declarative — loading it never touches the filesystem paths it
records. Whether a registered dataset's file actually exists is checked only when something
tries to load it (`adia.data.loader.load_dataset`), which keeps registration decoupled from
data availability. This is deliberate: it lets the system, benchmarks, and tests describe
"a dataset with this shape will exist here" before any real dataset has been acquired.

Example registration (the shape `DatasetConfig` expects, one entry per dataset in the JSON
file `load_registry` reads):

```json
[
  {
    "dataset_id": "ecommerce",
    "file_path": "data/processed/ecommerce.parquet",
    "description": "E-commerce order transactions: dates, categories, regions.",
    "target_columns": ["is_returned"]
  }
]
```
"""

from pathlib import Path

from pydantic import TypeAdapter

from adia.models.dataset import DatasetConfig

_registry_list_adapter: TypeAdapter[list[DatasetConfig]] = TypeAdapter(list[DatasetConfig])


def load_registry(path: str | Path) -> dict[str, DatasetConfig]:
    """Load a dataset registry file into an ID-keyed dict.

    Args:
        path: Path to a JSON file holding a list of dataset registrations.

    Returns:
        Registered datasets, keyed by `dataset_id`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        pydantic.ValidationError: If an entry doesn't match `DatasetConfig`.
        ValueError: If two entries share the same `dataset_id`.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset registry not found: {resolved}")

    configs = _registry_list_adapter.validate_json(resolved.read_bytes())
    registry: dict[str, DatasetConfig] = {}
    for config in configs:
        if config.dataset_id in registry:
            raise ValueError(f"Duplicate dataset_id in registry: '{config.dataset_id}'")
        registry[config.dataset_id] = config
    return registry


def get_dataset_config(registry: dict[str, DatasetConfig], dataset_id: str) -> DatasetConfig:
    """Look up one dataset's registration.

    Args:
        registry: A registry as returned by `load_registry`.
        dataset_id: The dataset to look up.

    Returns:
        The matching `DatasetConfig`.

    Raises:
        KeyError: If `dataset_id` is not registered.
    """
    try:
        return registry[dataset_id]
    except KeyError:
        raise KeyError(f"Dataset '{dataset_id}' is not registered.") from None


def save_registry(configs: list[DatasetConfig], path: str | Path) -> None:
    """Serialize a list of dataset registrations to a JSON file.

    Args:
        configs: The dataset registrations to persist.
        path: Destination file path; overwritten if it already exists.
    """
    Path(path).write_bytes(_registry_list_adapter.dump_json(configs, indent=2))
