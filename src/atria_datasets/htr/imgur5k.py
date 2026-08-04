from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from atria_core.datasets import Dataset, DatasetConfig
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import OCRAnnotation
from atria_core.types._generic._image import Image
from PIL import Image as PILImage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets
from atria_datasets.utils import require_manual_path

_HOMEPAGE = "https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset"
_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "val",
    DatasetSplitType.test: "test",
}


@datasets.register("imgur5k")
@pydantic_dataclass(frozen=True)
class IMGUR5KConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> IMGUR5K:
        return IMGUR5K(self, **kwargs)


class IMGUR5K(Dataset[IMGUR5KConfig, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        root = require_manual_path(
            data_dir,
            "imgur5k",
            homepage=_HOMEPAGE,
            instructions="Run upstream download_imgur5k.py with output_dir=imgur5k/images and keep its generated JSON files in imgur5k/dataset_info.",
        )
        return {"imgur5k": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="IMGUR5K in-the-wild English handwriting with rotated word boxes.",
            homepage=_HOMEPAGE,
            license="CC BY-NC 4.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> list[tuple[Path, OCRAnnotation]]:
        root = Path(data_dir) / "imgur5k"
        info_path = (
            root / "dataset_info" / f"imgur5k_annotations_{_SPLIT_NAMES[split]}.json"
        )
        data = json.loads(info_path.read_text(encoding="utf-8"))
        samples: list[tuple[Path, OCRAnnotation]] = []
        for index, metadata in data["index_id"].items():
            image_path = root / "images" / f"{index}.jpg"
            if not image_path.exists():
                candidate = Path(metadata["image_path"])
                image_path = candidate if candidate.is_absolute() else root / candidate
            if not image_path.exists():
                continue
            width, height = PILImage.open(image_path).size
            texts, bboxes, angles = [], [], []
            for annotation_id in data["index_to_ann_map"][index]:
                item = data["ann_id"][annotation_id]
                if item["bounding_box"] == ".":
                    continue
                xc, yc, box_width, box_height, angle = map(
                    float, item["bounding_box"].split(",")
                )
                texts.append(item["word"])
                bboxes.append(
                    [
                        (xc - box_width / 2) / width,
                        (yc - box_height / 2) / height,
                        (xc + box_width / 2) / width,
                        (yc + box_height / 2) / height,
                    ]
                )
                angles.append(angle)
            if texts:
                annotation = replace(
                    OCRAnnotation.from_words(texts, np.clip(bboxes, 0.0, 1.0)),
                    angles=np.asarray(angles),
                )
                samples.append((image_path, annotation))
        return samples

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        def transform(sample: tuple[Path, OCRAnnotation]) -> SinglePageDocumentInstance:
            image_path, annotation = sample
            return SinglePageDocumentInstance(
                sample_id=image_path.stem, visual=Image(file_path=str(image_path))
            ).add_annotation(annotation)

        return transform
