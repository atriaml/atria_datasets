from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import requests
import tqdm
from atria_core.datasets import AtriaDownloadManager, Dataset, UrlSpec
from atria_core.logger import get_logger
from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    DatasetMetadata,
    DatasetSplitType,
    MultiPageDocumentInstance,
    MultiPageQAPair,
    MultiPageQuestionAnsweringAnnotation,
    ObjectDetectionAnnotation,
    SinglePageDocumentInstance,
)

logger = get_logger(__name__)

_HOMEPAGE = "https://github.com/nttmdlab-nlp/SlideVQA"
_LICENSE = "CC BY-NC-SA 4.0"
_UPSTREAM_RAW = (
    "https://raw.githubusercontent.com/nttmdlab-nlp/SlideVQA/main/annotations"
)
_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "dev",
    DatasetSplitType.test: "test",
}
_DOWNLOAD_WORKERS = 8
_DOWNLOAD_TIMEOUT_SECONDS = 30


@dataclass
class _BBox:
    bbox_id: int
    label_name: str
    bbox: tuple[float, float, float, float]


@dataclass
class _QARecord:
    deck_name: str
    image_urls: list[str]
    qa_id: int
    question: str
    answer: str
    arithmetic_expression: str | None
    evidence_pages: list[int]


def _parse_qa_records(path: Path) -> list[_QARecord]:
    records: list[_QARecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data: dict[str, Any] = json.loads(line)
        arithmetic_expression: str = data["arithmetic_expression"]
        records.append(
            _QARecord(
                deck_name=data["deck_name"],
                image_urls=data["image_urls"],
                qa_id=data["qa_id"],
                question=data["question"],
                answer=data["answer"],
                arithmetic_expression=None
                if arithmetic_expression == "None"
                else arithmetic_expression,
                evidence_pages=list(data["evidence_pages"]),
            )
        )
    return records


def _parse_bboxes(path: Path) -> dict[str, dict[int, list[_BBox]]]:
    bboxes_by_deck: dict[str, dict[int, list[_BBox]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data: dict[str, Any] = json.loads(line)
        pages: dict[int, list[_BBox]] = {}
        for page_number, page_bboxes in data["bboxes"]:
            pages[page_number - 1] = [
                _BBox(
                    bbox_id=item["bbox_id"],
                    label_name=item["class"],
                    bbox=(
                        item["bbox"][0],
                        item["bbox"][1],
                        item["bbox"][2],
                        item["bbox"][3],
                    ),
                )
                for item in page_bboxes
            ]
        bboxes_by_deck[data["deck_name"]] = pages
    return bboxes_by_deck


def _download_deck_image(url: str, image_path: Path) -> bool:
    if image_path.exists():
        return True
    try:
        with requests.get(
            url=url,
            headers={"User-Agent": "Atria SlideVQA downloader"},
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            content = response.content
        if len(content) < 100:
            return False
        incomplete_path = image_path.with_suffix(image_path.suffix + ".incomplete")
        incomplete_path.write_bytes(content)
        incomplete_path.replace(image_path)
        return True
    except (OSError, requests.RequestException):
        return False


def _download_deck_images(records: list[_QARecord], images_dir: Path) -> None:
    decks = {record.deck_name: record.image_urls for record in records}
    download_items: list[tuple[str, Path]] = []
    for deck_name, image_urls in decks.items():
        deck_dir = images_dir / deck_name
        deck_dir.mkdir(parents=True, exist_ok=True)
        for page_number, url in enumerate(image_urls):
            download_items.append((url, deck_dir / f"{page_number}.jpg"))

    def download(item: tuple[str, Path]) -> bool:
        return _download_deck_image(*item)

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
        list(
            tqdm.tqdm(
                iterable=executor.map(download, download_items),
                total=len(download_items),
                desc="Downloading SlideVQA slide images",
                unit="image",
            )
        )


class InputTransform:
    def __init__(
        self, images_dir: Path, bboxes_by_deck: dict[str, dict[int, list[_BBox]]]
    ) -> None:
        self._images_dir = images_dir
        self._bboxes_by_deck = bboxes_by_deck

    def _build_page(
        self, deck_name: str, page_number: int, image_path: Path
    ) -> SinglePageDocumentInstance:
        page = SinglePageDocumentInstance.from_image(
            image_path, sample_id=f"{deck_name}#{page_number}"
        )
        page_bboxes = self._bboxes_by_deck.get(deck_name, {}).get(page_number)
        if not page_bboxes:
            return page
        annotation = replace(
            ObjectDetectionAnnotation.from_objects([
                AnnotatedObject(
                    label_value=index,
                    label_name=item.label_name,
                    bbox=np.asarray(item.bbox, dtype=np.float64),
                )
                for index, item in enumerate(page_bboxes)
            ]),
            bbox_mode=BoundingBoxMode.XYWH,
            normalized=False,
        )
        return page.add_annotation(annotation=annotation)

    def __call__(self, sample: _QARecord) -> MultiPageDocumentInstance:
        deck_dir = self._images_dir / sample.deck_name
        pages: list[SinglePageDocumentInstance] = []
        for page_number in range(len(sample.image_urls)):
            image_path = deck_dir / f"{page_number}.jpg"
            if image_path.exists():
                pages.append(
                    self._build_page(sample.deck_name, page_number, image_path)
                )
        qa_pair = MultiPageQAPair(
            id=sample.qa_id,
            question_text=sample.question,
            answer_text=sample.answer,
            evidence_pages=sample.evidence_pages,
            arithmetic_expression=sample.arithmetic_expression,
        )
        return MultiPageDocumentInstance(
            sample_id=f"{sample.deck_name}#{sample.qa_id}",
            pages=pages,
            metadata={"deck_name": sample.deck_name},
        ).add_annotation(
            annotation=MultiPageQuestionAnsweringAnnotation(qa_pairs=[qa_pair])
        )


class SlideVQA(Dataset[MultiPageDocumentInstance]):
    """SlideVQA: question answering over multi-page slide decks, with
    evidence-page and bounding-box ground truth."""

    __module_name__ = "slidevqa"

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        data_root = Path(data_dir)
        root = data_root / "slidevqa"
        annotations_dir = root / "annotations"
        images_dir = root / "images"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        manager = AtriaDownloadManager(
            data_dir=data_root, download_dir=data_root / ".download_cache"
        )
        manager.download_and_extract(
            data_urls=[
                UrlSpec(
                    url=f"{_UPSTREAM_RAW}/{annotation_type}/{split_name}.jsonl",
                    url_ext=".jsonl",
                    rel_output_file_path=(
                        f"slidevqa/annotations/{annotation_type}/{split_name}.jsonl"
                    ),
                )
                for annotation_type in ("qa", "bbox")
                for split_name in _SPLIT_NAMES.values()
            ],
            extract=False,
        )

        complete_marker = root / ".images_download_complete"
        if not complete_marker.exists():
            records = [
                record
                for split_name in _SPLIT_NAMES.values()
                for record in _parse_qa_records(
                    annotations_dir / "qa" / f"{split_name}.jsonl"
                )
            ]
            _download_deck_images(records=records, images_dir=images_dir)
            complete_marker.write_text(data="done\n", encoding="utf-8")

        return {"slidevqa": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "SlideVQA: question answering over multi-page slide decks, "
                "requiring multi-hop reasoning across evidence pages."
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> list[_QARecord]:
        root = Path(data_dir) / "slidevqa"
        return _parse_qa_records(
            root / "annotations" / "qa" / f"{_SPLIT_NAMES[split]}.jsonl"
        )

    def _build_input_transform(
        self,
    ) -> Callable[[Any], MultiPageDocumentInstance]:
        root = self.data_dir / "slidevqa"
        bboxes_by_deck: dict[str, dict[int, list[_BBox]]] = {}
        for split_name in _SPLIT_NAMES.values():
            bbox_path = root / "annotations" / "bbox" / f"{split_name}.jsonl"
            if bbox_path.exists():
                bboxes_by_deck.update(_parse_bboxes(bbox_path))
        return InputTransform(images_dir=root / "images", bboxes_by_deck=bboxes_by_deck)
