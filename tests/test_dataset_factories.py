from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, get_type_hints

import pytest
from atria_core.datasets import Dataset
from atria_core.types import DatasetSplitType
from atria_core.types._generic._elements import OCRLevel

import atria_datasets

FactoryCase = tuple[str, dict[str, Any], dict[str, Any]]


FACTORY_CASES: list[FactoryCase] = [
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


def test_factory_cases_cover_public_namespace() -> None:
    assert {case[0] for case in FACTORY_CASES} == set(atria_datasets.__all__)


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
    ("factory_name", "factory_kwargs", "expected_config"),
    FACTORY_CASES,
    ids=[case[0] for case in FACTORY_CASES],
)
def test_dataset_factory(
    factory_name: str,
    factory_kwargs: dict[str, Any],
    expected_config: dict[str, Any],
    stub_dataset_build: None,
) -> None:
    factory: Callable[..., Dataset[Any, Any]] = getattr(atria_datasets, factory_name)
    expected_type = get_type_hints(factory)["return"]

    dataset = factory(
        **factory_kwargs, access_token="test-token", split=DatasetSplitType.train
    )

    assert type(dataset) is expected_type
    assert dataset.data_dir.name == factory_name
    assert dataset._factory_test_build_args == (  # type: ignore[attr-defined]
        str(dataset.data_dir),
        DatasetSplitType.train,
        "test-token",
    )
    for field_name, expected_value in expected_config.items():
        assert getattr(dataset.config, field_name) == expected_value


def test_load_dataset_forwards_shared_and_specific_arguments(
    stub_dataset_build: None,
) -> None:
    dataset = atria_datasets.load_dataset(
        "iam",
        access_token="test-token",
        split=DatasetSplitType.validation,
        crop_to_handwriting=False,
    )

    assert dataset.config.crop_to_handwriting is False
    assert dataset._factory_test_build_args == (  # type: ignore[attr-defined]
        str(dataset.data_dir),
        DatasetSplitType.validation,
        "test-token",
    )
