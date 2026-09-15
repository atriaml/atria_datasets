from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path

from atria_core.datasets import Dataset

import atria_datasets


def test_registration_is_lazy() -> None:
    code = """
import sys
import atria_datasets

assert len(atria_datasets.datasets.list()) == 22
assert not any(
    name.startswith((
        "atria_datasets.cr.",
        "atria_datasets.htr.",
        "atria_datasets.qa.",
    ))
    for name in sys.modules
)

atria_datasets.datasets.get("squad")
assert "atria_datasets.qa.squad" in sys.modules
assert "atria_datasets.htr.iam" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_registered_name_resolves_to_a_dataset_subclass() -> None:
    assert len(atria_datasets.datasets.list()) == 22
    for _, dataset_cls in atria_datasets.datasets.items():
        assert issubclass(dataset_cls, Dataset)


def test_every_dataset_class_has_a_registered_name() -> None:
    package_root = Path(atria_datasets.__file__).parent
    for category in ("cr", "htr", "qa"):
        package = importlib.import_module(f"atria_datasets.{category}")
        for module_info in pkgutil.iter_modules([str(package_root / category)]):
            if not module_info.name.startswith("_"):
                importlib.import_module(f"{package.__name__}.{module_info.name}")

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
    registered_types = {dataset_cls for _, dataset_cls in atria_datasets.datasets.items()}
    assert dataset_types == registered_types
