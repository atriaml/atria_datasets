from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

from atria_core.datasets import Dataset

import atria_datasets


def test_import_is_lazy() -> None:
    code = """
import sys
import atria_datasets

assert len(atria_datasets.__all__) == 21
assert not any(
    name.startswith((
        "atria_datasets.cr.",
        "atria_datasets.htr.",
        "atria_datasets.qa.",
    ))
    for name in sys.modules
)

assert callable(atria_datasets.squad)
assert "atria_datasets.qa.squad" in sys.modules
assert "atria_datasets.htr.iam" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_export_resolves_to_a_factory() -> None:
    assert len(atria_datasets.__all__) == 21
    assert all(
        callable(getattr(atria_datasets, name)) for name in atria_datasets.__all__
    )


def test_every_dataset_class_has_an_exported_factory() -> None:
    package_root = Path(atria_datasets.__file__).parent
    for category in ("cr", "htr", "qa"):
        package = importlib.import_module(f"atria_datasets.{category}")
        for module_info in pkgutil.iter_modules([str(package_root / category)]):
            if not module_info.name.startswith("_"):
                importlib.import_module(f"{package.__name__}.{module_info.name}")

    factory_return_types = {
        get_type_hints(getattr(atria_datasets, name))["return"]
        for name in atria_datasets.__all__
    }
    dataset_types = {
        value
        for module_name, module in tuple(sys.modules.items())
        if module_name.startswith(
            ("atria_datasets.cr.", "atria_datasets.htr.", "atria_datasets.qa.")
        )
        and not module_name.rsplit(".", 1)[-1].startswith("_")
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module_name
        and issubclass(value, Dataset)
    }

    assert dataset_types == factory_return_types
