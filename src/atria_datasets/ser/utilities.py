"""Helpers shared by the semantic entity recognition datasets."""

import numpy as np
import pandas as pd

from atria_core.types import BoundingBoxMode, DocumentContent, ElementArray, OCRLevel


def _get_line_bboxes(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    x = [bboxes[i][j] for i in range(len(bboxes)) for j in range(0, len(bboxes[i]), 2)]
    y = [bboxes[i][j] for i in range(len(bboxes)) for j in range(1, len(bboxes[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxes))]
    return bbox


def _normalize_bbox(
    bbox: tuple[int, int, int, int], size: tuple[int, int]
) -> list[int]:
    return [
        float(bbox[0] / size[0]),
        float(bbox[1] / size[1]),
        float(bbox[2] / size[0]),
        float(bbox[3] / size[1]),
    ]


def _get_sorted_indices(df: pd.DataFrame) -> pd.DataFrame:
    def is_same_line(word1, word2):
        return abs(word1["cy"] - word2["cy"]) <= word1["bbox_threshold"]

    df = df.sort_values(by=["cy", "cx"]).reset_index(drop=True)
    lines = []
    current_line = []
    for i in range(len(df)):
        if i == 0:
            current_line.append(df.iloc[i])
        else:
            prev_word = df.iloc[i - 1]
            current_word = df.iloc[i]
            if is_same_line(prev_word, current_word):
                current_line.append(current_word)
            else:
                lines.append(current_line)
                current_line = [current_word]

    if current_line:
        lines.append(current_line)

    sorted_lines = [pd.DataFrame(line).sort_values(by="cx") for line in lines]
    sorted_df = pd.concat(sorted_lines).reset_index(drop=True)
    return [int(x) for x in sorted_df["index"].tolist()]


def _sorted_indices_in_reading_order(
    word_bboxes: list[tuple[int, int, int, int]], bbox_threshold: float = 0.4
) -> dict:
    word_coords = []
    for idx, word_bbox in enumerate(word_bboxes):
        word_coords.append(
            {
                "index": idx,
                "cx": word_bbox[0],
                "cy": word_bbox[1],
                "bbox_threshold": (word_bbox[3] - word_bbox[1]) * bbox_threshold,
            }
        )
    df = pd.DataFrame(word_coords)
    return _get_sorted_indices(df)


def content_from_words_and_segments(
    words: list[str],
    boxes: list[list[float]],
    segment_ids: list[int],
) -> DocumentContent:
    """Build line-and-word document content from word boxes and their segment ids.

    Each segment becomes a line element spanning its words' boxes, and each
    word becomes a child element of its segment.
    """
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
        # Annotated quads can extend slightly past the page; ElementArray
        # requires normalized coordinates in [0, 1].
        bboxes=np.clip(
            np.asarray(segment_boxes + boxes, dtype=np.float64).reshape(-1, 4),
            a_min=0.0,
            a_max=1.0,
        ),
        bbox_mode=BoundingBoxMode.XYXY,
        normalized=True,
        texts=np.asarray([""] * segment_count + words, dtype=object),
    )
    return DocumentContent(elements=elements)
