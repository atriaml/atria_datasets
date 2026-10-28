"""Helpers shared by the COCO-style Hugging Face layout-analysis datasets."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from datasets import Dataset as HFDataset
from datasets import IterableDataset

from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    DatasetSplitType,
    LayoutAnalysisAnnotation,
)
from atria_datasets.layout_analysis.config import LayoutDatasetConfig


def sample_cap_for_split(
    config: LayoutDatasetConfig, split: DatasetSplitType
) -> int | None:
    """Return the configured sample cap for `split`, or `None` if uncapped."""
    return {
        DatasetSplitType.train: config.max_train_samples,
        DatasetSplitType.validation: config.max_validation_samples,
        DatasetSplitType.test: config.max_test_samples,
    }.get(split)


def limit_split(
    iterator: HFDataset | IterableDataset, limit: int | None
) -> HFDataset | IterableDataset:
    """Keep at most `limit` leading rows of a streamed or materialized split."""
    if limit is None:
        return iterator
    if isinstance(iterator, IterableDataset):
        return iterator.take(limit)
    return iterator.select(range(min(len(iterator), limit)))


def annotated_objects(
    annotations: list[dict[str, Any]],
    class_names: list[str],
    *,
    category_offset: int,
    width: int,
    height: int,
) -> list[AnnotatedObject]:
    """Convert COCO-style annotations into normalized layout objects.

    Ignored, empty and sub-pixel boxes are dropped, as are categories outside
    `class_names` once shifted by `category_offset`. Only single-polygon
    segmentations are kept.
    """
    result: list[AnnotatedObject] = []
    for annotation in annotations:
        if annotation.get("ignore", False):
            continue
        x, y, box_width, box_height = annotation["bbox"]
        if annotation["area"] <= 0 or box_width < 1 or box_height < 1:
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


def layout_annotation(objects: list[AnnotatedObject]) -> LayoutAnalysisAnnotation:
    """Build a normalized XYXY layout annotation, empty when `objects` is empty."""
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
    return replace(layout, bbox_mode=BoundingBoxMode.XYXY, normalized=True)
