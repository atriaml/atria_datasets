from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, UrlSpec
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    Image,
    OCRLevel,
    SinglePageDocumentInstance,
    TranscriptionAnnotation,
)
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset


@lru_cache(maxsize=1)
def _load_rimes() -> DatasetDict:
    return load_dataset("Teklia/RIMES-2011-line")


class RIMES(Dataset[SinglePageDocumentInstance]):
    """RIMES-2011 line images and transcriptions from Hugging Face."""

    __module_name__ = "rimes"

    def _download_urls(self) -> list[UrlSpec]:
        return []

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="RIMES-2011 line images with French transcriptions.",
            homepage="https://huggingface.co/datasets/Teklia/RIMES-2011-line",
            license="See the source dataset terms.",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType(split) for split in _load_rimes().keys()]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> HFDataset:
        dataset = _load_rimes()[split.value]
        return dataset.add_column(
            "sample_id", [f"{split.value}-{index:05d}" for index in range(len(dataset))]
        )

    def _build_input_transform(
        self,
    ) -> Callable[[dict[str, Any]], SinglePageDocumentInstance]:
        def transform(row: dict[str, Any]) -> SinglePageDocumentInstance:
            sample_id = row["sample_id"]
            image_path = self.data_dir / "line_images" / f"{sample_id}.png"
            if not image_path.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                row["image"].save(image_path)
            return SinglePageDocumentInstance(
                sample_id=sample_id,
                visual=Image(file_path=str(image_path)),
            ).add_annotation(
                annotation=TranscriptionAnnotation(
                    text=row["text"], level=OCRLevel.line
                )
            )

        return transform
