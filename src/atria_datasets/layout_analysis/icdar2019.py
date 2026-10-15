"""ICDAR 2019 Table Detection Challenge (cTDaR), track A modern split."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

import numpy as np

from atria_core.datasets import Dataset, DatasetConfig, UrlSpec
from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    Image,
    LayoutAnalysisAnnotation,
    SinglePageDocumentInstance,
)

_CITATION = """\
@inproceedings{icdar2019,
    title={ICDAR2019 Competition on Table Detection and Recognition},
    author={Zhang, Yifan and Wang, Yuhui and Zhang, Yifan and Liu, Zhen and Li, Yujie and Wang, Xiaoyang},
    booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
    pages={1--6},
    year={2019},
    organization={IEEE}
}
"""
_DESCRIPTION = "The ICDAR 2019 Table Detection Challenge (cTDaR) focuses on detecting tables in document images."
_HOMEPAGE = "https://github.com/cndplab-founder/ICDAR2019_cTDaR/tree/master/samples"
_DATA_URL = "https://drive.google.com/file/d/1ES4TcZ4pU5-3mtSx7gQw49YNwgQjGH1n/view?usp=sharing"
_CLASSES = ["table"]
_SOURCE_CONFIG = "trackA_modern"
_SPLITS = (DatasetSplitType.train, DatasetSplitType.test)


@dataclass(frozen=True)
class _Icdar2019Record:
    file_name: str
    width: int
    height: int
    annotations: list[dict[str, Any]]


class _Icdar2019Split(Sequence[_Icdar2019Record]):
    def __init__(self, data_dir: str | Path, split: DatasetSplitType) -> None:
        base_dir = Path(data_dir) / "data" / "icdar2019" / _SOURCE_CONFIG
        image_dir = base_dir / split.value
        annotation_file = base_dir / f"{split.value}.json"
        if not image_dir.exists():
            raise FileNotFoundError(f"Split {split.value} does not exist in {base_dir}")

        with annotation_file.open(encoding="utf-8") as stream:
            coco = json.load(stream)

        categories = sorted(coco["categories"], key=lambda item: item["id"])
        category_names = [category["name"] for category in categories]
        if category_names != _CLASSES:
            raise AssertionError(
                f"Category names do not match between dataset config and json file. "
                f"Required: {_CLASSES}, found: {category_names}"
            )
        category_ids = {
            category["id"]: index for index, category in enumerate(categories)
        }
        images = sorted(coco["images"], key=lambda item: item["id"])
        image_ids = {image["id"] for image in images}
        annotation_ids = [annotation["id"] for annotation in coco["annotations"]]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise AssertionError(f"Annotation ids in '{annotation_file}' are not unique!")

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in image_ids:
                continue
            if annotation.get("ignore", 0) != 0:
                raise AssertionError('"ignore" in COCO json file is not supported.')

            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, list):
                segmentation = [
                    polygon
                    for polygon in segmentation
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if not segmentation:
                    continue

            annotations_by_image[image_id].append(
                {**annotation, "segmentation": segmentation}
            )

        self.samples = [
            _Icdar2019Record(
                file_name=str(image_dir / image["file_name"]),
                width=image["width"],
                height=image["height"],
                annotations=[
                    {
                        **annotation,
                        "category_id": category_ids[annotation["category_id"]],
                    }
                    for annotation in annotations_by_image[image["id"]]
                ],
            )
            for image in images
        ]

    @overload
    def __getitem__(self, index: int) -> _Icdar2019Record: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_Icdar2019Record]: ...

    def __getitem__(
        self, index: int | slice
    ) -> _Icdar2019Record | Sequence[_Icdar2019Record]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class _Icdar2019Transform:
    def __call__(self, sample: _Icdar2019Record) -> SinglePageDocumentInstance:
        annotated_objects = [
            AnnotatedObject(
                label_value=annotation["category_id"],
                label_name=_CLASSES[annotation["category_id"]],
                bbox=_normalized_xyxy(
                    annotation["bbox"], width=sample.width, height=sample.height
                ),
                segmentation=_normalized_segmentation(
                    annotation.get("segmentation"),
                    width=sample.width,
                    height=sample.height,
                ),
                iscrowd=bool(annotation["iscrowd"]),
            )
            for annotation in sample.annotations
        ]
        layout_annotation = (
            LayoutAnalysisAnnotation.from_objects(annotated_objects)
            if annotated_objects
            else LayoutAnalysisAnnotation(
                label_values=np.empty(0, dtype=np.int64),
                label_names=np.empty(0, dtype=object),
                bboxes=np.empty((0, 4), dtype=np.float64),
                iscrowd=np.empty(0, dtype=np.bool_),
            )
        )
        layout_annotation = replace(
            layout_annotation,
            bbox_mode=BoundingBoxMode.XYXY,
            normalized=True,
        )
        return SinglePageDocumentInstance(
            sample_id=Path(sample.file_name).name,
            visual=Image(file_path=sample.file_name),
        ).add_annotation(
            layout_annotation
        )


def _normalized_xyxy(
    bbox: list[float], *, width: int, height: int
) -> np.ndarray:
    x, y, box_width, box_height = bbox
    return np.asarray(
        [x / width, y / height, (x + box_width) / width, (y + box_height) / height],
        dtype=np.float64,
    )


def _normalized_segmentation(
    segmentation: Any, *, width: int, height: int
) -> np.ndarray | None:
    if not isinstance(segmentation, list) or len(segmentation) != 1:
        return None
    points = np.asarray(segmentation[0], dtype=np.float64).reshape(-1, 2)
    points[:, 0] /= width
    points[:, 1] /= height
    return points


class Icdar2019(Dataset[SinglePageDocumentInstance, DatasetConfig]):
    """ICDAR 2019 cTDaR track A modern table layout annotations."""

    __module_name__ = "icdar2019"

    def _download_urls(self) -> list[UrlSpec]:
        return [
            UrlSpec(
                url=_DATA_URL,
                url_ext=".zip",
                rel_output_file_path="data.zip",
            )
        ]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license="Apache-2.0 license",
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLITS)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> _Icdar2019Split:
        return _Icdar2019Split(data_dir=data_dir, split=split)

    def _build_input_transform(
        self,
    ) -> Callable[[Any], SinglePageDocumentInstance]:
        return _Icdar2019Transform()
