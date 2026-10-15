"""Hugging Face layout-analysis datasets used by DocDjinn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from atria_core.datasets import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    Image,
    LayoutAnalysisAnnotation,
    SinglePageDocumentInstance,
)


class _LayoutConfig(HuggingfaceDatasetConfig):
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


def _cap_for_split(config: _LayoutConfig, split: DatasetSplitType) -> int | None:
    return {
        DatasetSplitType.train: config.max_train_samples,
        DatasetSplitType.validation: config.max_validation_samples,
        DatasetSplitType.test: config.max_test_samples,
    }.get(split)


def _objects(
    annotations: list[dict[str, Any]],
    class_names: list[str],
    *,
    category_offset: int,
    width: int,
    height: int,
) -> list[AnnotatedObject]:
    result = []
    for annotation in annotations:
        if annotation.get("ignore", False):
            continue
        x, y, box_width, box_height = annotation["bbox"]
        if (
            annotation["area"] <= 0
            or box_width < 1
            or box_height < 1
        ):
            continue
        category = annotation["category_id"] - category_offset
        if category < 0 or category >= len(class_names):
            continue
        bbox = np.asarray(
            [x / width, y / height, (x + box_width) / width, (y + box_height) / height],
            dtype=np.float64,
        )
        polygons = annotation.get("segmentation")
        segmentation = None
        if isinstance(polygons, list) and len(polygons) == 1:
            polygon = np.asarray(polygons[0], dtype=np.float64)
            if polygon.size >= 6 and polygon.size % 2 == 0:
                segmentation = polygon.reshape(-1, 2)
                segmentation[:, 0] /= width
                segmentation[:, 1] /= height
        result.append(
            AnnotatedObject(
                label_value=category,
                label_name=class_names[category],
                bbox=bbox,
                segmentation=segmentation,
                iscrowd=bool(annotation["iscrowd"]),
            )
        )
    return result


class DocLayNet(HuggingfaceDataset[SinglePageDocumentInstance, _LayoutConfig]):
    """DocLayNet with its original document and layout annotations."""

    __module_name__ = "doclaynet"
    __hf_repo__ = "ds4sd/DocLayNet"
    __hf_config_name__ = "2022.08"

    _DOC_CLASSES = [
        "financial_reports",
        "scientific_articles",
        "laws_and_regulations",
        "government_tenders",
        "manuals",
        "patents",
    ]
    _LAYOUT_CLASSES = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title ",
    ]

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        iterator = super()._build_split_iterator(split, data_dir)
        limit = _cap_for_split(self.config, split)
        return (
            iterator.select(range(min(len(iterator), limit)))
            if limit is not None
            else iterator
        )

    def _metadata(self) -> DatasetMetadata:
        return replace(
            super()._metadata(),
            dataset_labels=DatasetLabels(
                classification=self._DOC_CLASSES, layout=self._LAYOUT_CLASSES
            ),
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        def transform(sample: dict[str, Any]) -> SinglePageDocumentInstance:
            image = Image(content=sample["image"])
            objects = _objects(
                sample["objects"],
                self._LAYOUT_CLASSES,
                category_offset=0,
                width=image.width,
                height=image.height,
            )
            layout = (
                LayoutAnalysisAnnotation.from_objects(objects)
                if objects
                else LayoutAnalysisAnnotation(
                    label_values=np.empty(0, dtype=np.int64),
                    label_names=np.empty(0, dtype=object),
                    bboxes=np.empty((0, 4), dtype=np.float64),
                    iscrowd=np.empty(0, dtype=np.bool_),
                )
            )
            layout = replace(layout, bbox_mode=BoundingBoxMode.XYXY, normalized=True)
            doc_category = sample["doc_category"]
            return SinglePageDocumentInstance(
                sample_id=f"sample_{sample['image_id']}", visual=image
            ).add_annotation(
                ClassificationAnnotation(
                    label_value=self._DOC_CLASSES.index(doc_category),
                    label_name=doc_category,
                )
            ).add_annotation(layout)

        return transform


class PubLayNet(HuggingfaceDataset[SinglePageDocumentInstance, _LayoutConfig]):
    """PubLayNet with the original 4k training subset and layout labels."""

    __module_name__ = "publaynet"
    __hf_repo__ = "jordanparker6/publaynet"
    __hf_config_name__ = "default"
    _CLASSES = ["text", "title", "list", "table", "figure"]

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        iterator = super()._build_split_iterator(split, data_dir)
        limit = _cap_for_split(self.config, split)
        return (
            iterator.select(range(min(len(iterator), limit)))
            if limit is not None
            else iterator
        )

    def _metadata(self) -> DatasetMetadata:
        return replace(
            super()._metadata(), dataset_labels=DatasetLabels(layout=self._CLASSES)
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        def transform(sample: dict[str, Any]) -> SinglePageDocumentInstance:
            image = Image(content=sample["image"])
            objects = _objects(
                sample["annotations"],
                self._CLASSES,
                category_offset=1,
                width=image.width,
                height=image.height,
            )
            layout = (
                LayoutAnalysisAnnotation.from_objects(objects)
                if objects
                else LayoutAnalysisAnnotation(
                    label_values=np.empty(0, dtype=np.int64),
                    label_names=np.empty(0, dtype=object),
                    bboxes=np.empty((0, 4), dtype=np.float64),
                    iscrowd=np.empty(0, dtype=np.bool_),
                )
            )
            layout = replace(layout, bbox_mode=BoundingBoxMode.XYXY, normalized=True)
            return SinglePageDocumentInstance(
                sample_id=str(sample["id"]), visual=image
            ).add_annotation(layout)

        return transform
