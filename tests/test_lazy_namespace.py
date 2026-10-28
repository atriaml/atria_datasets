from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path

from atria_core.datasets import Dataset

import atria_datasets
from atria_datasets import _DATASET_IMPORT_PATHS

DATASET_CATEGORIES = (
    "cr",
    "document_classification",
    "htr",
    "layout_analysis",
    "qa",
    "ser",
)
DATASET_MODULE_PREFIXES = tuple(
    f"atria_datasets.{category}." for category in DATASET_CATEGORIES
)


def test_registration_is_lazy() -> None:
    code = f"""
import sys
import atria_datasets

assert len(atria_datasets.datasets.list()) == {len(_DATASET_IMPORT_PATHS)}
assert not any(name.startswith({DATASET_MODULE_PREFIXES!r}) for name in sys.modules)

atria_datasets.datasets.get("squad")
assert "atria_datasets.qa.squad" in sys.modules
assert "atria_datasets.htr.iam" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_registered_name_resolves_to_a_dataset_subclass() -> None:
    assert len(atria_datasets.datasets.list()) == len(_DATASET_IMPORT_PATHS)
    for _, dataset_cls in atria_datasets.datasets.items():
        assert issubclass(dataset_cls, Dataset)


def test_every_dataset_class_has_a_registered_name() -> None:
    package_root = Path(atria_datasets.__file__).parent
    for category in DATASET_CATEGORIES:
        package = importlib.import_module(f"atria_datasets.{category}")
        for module_info in pkgutil.iter_modules([str(package_root / category)]):
            if not module_info.name.startswith("_"):
                importlib.import_module(f"{package.__name__}.{module_info.name}")

    dataset_types = {
        value
        for module_name, module in tuple(sys.modules.items())
        if module_name.startswith(DATASET_MODULE_PREFIXES)
        and not module_name.rsplit(".", 1)[-1].startswith("_")
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module_name
        and issubclass(value, Dataset)
        and not vars(value).get("__abstract__", False)
    }
    registered_types = {dataset_cls for _, dataset_cls in atria_datasets.datasets.items()}
    assert dataset_types == registered_types
