"""Adapters for the per-word and per-segment fields used by the old loaders."""

from __future__ import annotations

import numpy as np

from atria_core.types import BoundingBoxMode, DocumentContent, ElementArray, OCRLevel


def content_from_words_and_segments(
    words: list[str],
    boxes: list[list[float]],
    segment_ids: list[int],
) -> DocumentContent:
    """Represent old word boxes and repeated segment boxes in ElementArray."""
    segment_count = max(segment_ids, default=-1) + 1
    segment_boxes = [
        [
            min(boxes[i][0] for i, segment_id in enumerate(segment_ids) if segment_id == group),
            min(boxes[i][1] for i, segment_id in enumerate(segment_ids) if segment_id == group),
            max(boxes[i][2] for i, segment_id in enumerate(segment_ids) if segment_id == group),
            max(boxes[i][3] for i, segment_id in enumerate(segment_ids) if segment_id == group),
        ]
        for group in range(segment_count)
    ]

    segment_ids_array = np.arange(segment_count, dtype=np.int64)
    word_ids = np.arange(segment_count, segment_count + len(words), dtype=np.int64)
    elements = ElementArray(
        ids=np.concatenate((segment_ids_array, word_ids)),
        parent_ids=np.concatenate(
            (
                np.full(segment_count, -1, dtype=np.int64),
                np.asarray(segment_ids, dtype=np.int64),
            )
        ),
        levels=np.concatenate(
            (
                np.full(segment_count, OCRLevel.line.value, dtype=np.int64),
                np.full(len(words), OCRLevel.word.value, dtype=np.int64),
            )
        ),
        bboxes=np.asarray(segment_boxes + boxes, dtype=np.float64).reshape(-1, 4),
        bbox_mode=BoundingBoxMode.XYXY,
        normalized=True,
        texts=np.asarray([""] * segment_count + words, dtype=object),
    )
    return DocumentContent(elements=elements)
