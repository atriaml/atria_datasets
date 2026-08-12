from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import IMAGE_SUFFIXES, TextAnnotationTransform
from atria_datasets.registry import datasets
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://www.nist.gov/srd/nist-special-database-19"


@datasets.register(name="nist_sd19")
@pydantic_dataclass(frozen=True)
class NISTSD19Config(DatasetConfig):
    def build_module(self, **kwargs: Any) -> NISTSD19:
        return NISTSD19(config=self, **kwargs)


def _class_name(directory: str) -> str:
    """SD19 by_class directories use hexadecimal Unicode/ASCII values."""
    try:
        return chr(int(directory, 16))
    except (ValueError, OverflowError):
        return directory


class NISTSD19(Dataset[NISTSD19Config, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = require_manual_path(
            data_dir=data_dir,
            expected_path="by_class",
            homepage=_HOMEPAGE,
            instructions="Obtain the licensed second-edition by_class.zip from NIST and extract it here.",
        )
        return {"by_class": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="NIST Special Database 19 second-edition isolated handprinted character images.",
            homepage=_HOMEPAGE,
            license="NIST licensed data; access is restricted and may require a fee",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[tuple[Path, str]]:
        root = Path(data_dir) / "by_class"

        def iterator() -> Iterable[tuple[Path, str]]:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    yield path, _class_name(directory=path.relative_to(root).parts[0])

        return iterator()

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return TextAnnotationTransform()
