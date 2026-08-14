from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from atria_core.datasets import Dataset
from atria_core.types import DatasetSplitType
from atria_core.types._generic._elements import OCRLevel

import atria_datasets
from atria_datasets import _DATASET_IMPORT_PATHS

DatasetCase = tuple[str, dict[str, Any], dict[str, Any]]


DATASET_CASES: list[DatasetCase] = [
    ("nist_sd19", {}, {}),
    ("austrian_newspapers", {}, {}),
    ("bentham", {}, {}),
    ("bullinger", {}, {}),
    ("cvl", {}, {}),
    ("fhswf_german_handwriting", {"config_name": "custom"}, {"config_name": "custom"}),
    ("german_kurrent_19c", {"license_subset": "all"}, {"license_subset": "all"}),
    ("gnhk", {}, {}),
    (
        "iam",
        {"include_bad_segmentations": True, "crop_to_handwriting": False},
        {"include_bad_segmentations": True, "crop_to_handwriting": False},
    ),
    ("iam_histdb", {"collection": "parzival"}, {"collection": "parzival"}),
    ("icdar2017_read_htr_a", {}, {}),
    ("icdar2017_read_htr_b", {}, {}),
    ("imgur5k", {}, {}),
    ("koenigsfelden", {"collection": "u17"}, {"collection": "u17"}),
    ("read_konzilsprotokolle", {}, {}),
    ("read_bozen", {}, {}),
    ("scadsai_german_handwriting", {"level": OCRLevel.word}, {"level": OCRLevel.word}),
    ("scadsai_german_fullpage", {}, {}),
    ("stabs_ratsbuecher_o10", {}, {}),
    ("squad", {}, {}),
    ("textvqa", {}, {}),
]


def _registered_class(name: str) -> type[Dataset[Any, Any]]:
    module_name, _, attribute = _DATASET_IMPORT_PATHS[name].rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


def test_dataset_cases_cover_registered_names() -> None:
    assert {case[0] for case in DATASET_CASES} == set(atria_datasets.datasets.list())


@pytest.fixture
def stub_dataset_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Skip data access while retaining the real constructor and config path."""

    def validate_data_dir(data_dir: str | Path) -> str:
        return str(tmp_path / Path(data_dir).name)

    def build_splits(
        self: Dataset[Any, Any],
        data_dir: str,
        split: DatasetSplitType | None,
        access_token: str | None,
    ) -> None:
        self._factory_test_build_args = (data_dir, split, access_token)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "atria_core.datasets._dataset._validate_data_dir", validate_data_dir
    )
    monkeypatch.setattr(Dataset, "_build_split_iterators", build_splits)
    monkeypatch.setattr(Dataset, "_persist_snapshot", lambda self: None)


@pytest.mark.parametrize(
    ("dataset_name", "dataset_kwargs", "expected_config"),
    DATASET_CASES,
    ids=[case[0] for case in DATASET_CASES],
)
def test_dataset_registry_create(
    dataset_name: str,
    dataset_kwargs: dict[str, Any],
    expected_config: dict[str, Any],
    stub_dataset_build: None,
) -> None:
    expected_type = _registered_class(dataset_name)

    dataset = atria_datasets.datasets.create(
        dataset_name,
        **dataset_kwargs,
        access_token="test-token",
        split=DatasetSplitType.train,
    )

    assert type(dataset) is expected_type
    assert dataset.data_dir.name == dataset_name
    assert dataset._factory_test_build_args == (  # type: ignore[attr-defined]
        str(dataset.data_dir),
        DatasetSplitType.train,
        "test-token",
    )
    for field_name, expected_value in expected_config.items():
        assert getattr(dataset.config, field_name) == expected_value


def test_dataset_registry_create_forwards_shared_and_specific_arguments(
    stub_dataset_build: None,
) -> None:
    dataset = atria_datasets.datasets.create(
        "iam",
        access_token="test-token",
        split=DatasetSplitType.validation,
        crop_to_handwriting=False,
    )

    config: Any = dataset.config
    assert config.crop_to_handwriting is False
    assert dataset._factory_test_build_args == (  # type: ignore[attr-defined]
        str(dataset.data_dir),
        DatasetSplitType.validation,
        "test-token",
    )
