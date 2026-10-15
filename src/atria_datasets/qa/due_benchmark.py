"""DUE Benchmark extractive QA datasets in the original DocDjinn format."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import PIL.Image
from pdf2image import convert_from_path

from atria_core.datasets import Dataset, DatasetConfig, UrlSpec
from atria_core.logger import get_logger
from atria_core.types import (
    BoundingBoxMode,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    Image,
    QAPair,
    QuestionAnsweringAnnotation,
    SinglePageDocumentInstance,
)

logger = get_logger(__name__)

_DATA_URLS = {
    "datasets": "https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/{name}.tar.gz",
    "pdfs": "https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/{name}.tar.gz",
}
_SPLITS = (DatasetSplitType.train, DatasetSplitType.test, DatasetSplitType.validation)
_HF_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "dev",
    DatasetSplitType.test: "test",
}
PIL.Image.MAX_IMAGE_PIXELS = 933120000


class DueBenchmarkConfig(DatasetConfig):
    """Options matching the original DUE loader defaults."""

    config_name: str
    ocr_engine: str = "microsoft_cv"
    answers_extraction_method: str = "v1"
    image_dpi: int = 72


def _clean(text: str) -> str:
    return re.sub(r"[,.;:()\-/$%&*]", "", text).lower()


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for col, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col] + 1,
                    previous[col - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _reverse_harsh_find(
    answer_tokens: list[str], words: list[str]
) -> tuple[int | None, int | None, str | None]:
    answer_raw = "".join(answer_tokens)
    answer = "".join(answer_tokens)
    for index, word in enumerate(words):
        if not answer_raw.endswith(word):
            continue
        joined = word
        for increment in range(1, 30):
            if index - increment < 0:
                break
            joined = words[index - increment] + joined
            distance = 3 if len(answer_raw) >= 15 else 1 if len(answer_raw) >= 5 else 0
            start, end = index - increment, index
            extracted = " ".join(words[start : end + 1])
            if _edit_distance(answer.replace(" ", ""), extracted.replace(" ", "")) <= distance:
                return start, end, extracted
    return None, None, None


def _harsh_find(
    answer_tokens: list[str], words: list[str]
) -> tuple[int | None, int | None, str | None]:
    answer_raw = "".join(answer_tokens)
    answer = " ".join(answer_tokens)
    if len(answer_tokens) == 1:
        for index, word in enumerate(words):
            distance = 0 if len(answer) < 5 else 1
            if _edit_distance(answer, word) <= distance:
                return index, index, word
    for index, word in enumerate(words):
        if not answer_raw.startswith(word):
            continue
        joined = word
        for increment in range(1, 30):
            if index + increment >= len(words):
                break
            joined += words[index + increment]
            distance = 1 if len(answer_raw) >= 5 else 0
            start, end = index, index + increment
            extracted = " ".join(words[start : end + 1])
            if _edit_distance(answer.replace(" ", ""), extracted.replace(" ", "")) <= distance:
                return start, end, extracted
    return _reverse_harsh_find(answer_tokens, words)


def _get_answer_indices(
    words: list[str], answer: str
) -> tuple[int | None, int | None, str | None]:
    answer_tokens = [_clean(token) for token in answer.split()]
    words = [_clean(word) for word in words]
    answer_compact = " ".join(answer_tokens)
    if not answer_tokens:
        return None, None, None
    start = words.index(answer_tokens[0]) if answer_tokens[0] in words else None
    end = words.index(answer_tokens[-1]) if answer_tokens[-1] in words else None
    if start is None or end is None:
        return _harsh_find(answer_tokens, words)
    if start > end:
        if answer_tokens[-1] in words[start:]:
            end = words[start:].index(answer_tokens[-1]) + start
        else:
            return _harsh_find(answer_tokens, words)
    extracted = " ".join(words[start : end + 1])
    if answer_compact.replace(" ", "") != extracted.replace(" ", ""):
        return _harsh_find(answer_tokens, words)
    return start, end, extracted


def _answer_span(words: list[str], answer: str) -> tuple[int, int, str]:
    start, end, extracted = _get_answer_indices(words, answer)
    answer_clean = _clean(answer)
    extracted_clean = _clean(extracted) if extracted is not None else None
    distance = (
        _edit_distance(extracted_clean.replace(" ", ""), answer_clean.replace(" ", ""))
        if extracted_clean is not None
        else 1000
    )
    if start == -1 or start is None or distance > 5:
        start = end = -1
    if start == -1 or extracted_clean is None or len(extracted_clean) > 150 or not extracted_clean:
        return -1, -1, ""
    return int(start), int(end), extracted_clean


def _annotations_for_document(doc: dict[str, Any]) -> Iterator[tuple[str, list[str]]]:
    for annotation in doc.get("annotations", []):
        question = annotation["key"]
        values: list[str] = []
        child_values: list[tuple[str, list[str]]] = []
        for value in annotation.get("values", []):
            if "children" in value:
                for child in value["children"]:
                    answers = [entry["value"] for entry in child["values"]]
                    if answers:
                        child_values.append(
                            (
                                f"What are the {question} values for the {child['key']} column?",
                                answers,
                            )
                        )
            elif "value_variants" in value:
                values.extend(value["value_variants"])
            else:
                values.append(value["value"])
        yield from child_values
        if values:
            yield question, values


class _DueSplit:
    def __init__(self, data_dir: Path, config: DueBenchmarkConfig, split: DatasetSplitType):
        self._data_dir = data_dir
        self._config = config
        self._split = split
        self._name = config.config_name[2:]
        self._extracted_name = {
            "InfographicsVQA": "infographics_vqa",
            "DocVQA": "docvqa",
            "PWC": "AxCell",
            "KleisterCharity": "kleister-charity",
        }.get(self._name, self._name)
        self._directory = (
            data_dir / "datasets" / self._name / "aws_neurips_time" / self._extracted_name
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        split_dir = self._directory / _HF_SPLIT_NAMES[self._split]
        docs_path = split_dir / "document.jsonl"
        content_path = split_dir / "documents_content.jsonl"
        grouped: list[dict[str, Any]] = []

        def flush_group() -> dict[str, Any] | None:
            if not grouped:
                return None
            first = grouped[0]
            pdf_path = (
                self._data_dir
                / "pdfs"
                / self._name
                / self._name
                / first["sample_id"]
            )
            if pdf_path.suffix.lower() != ".pdf":
                pdf_path = pdf_path.with_suffix(".pdf")
            if not pdf_path.exists():
                logger.warning("File %s not found. Skipping it", pdf_path)
                return None
            return {**first, "pdf_path": pdf_path, "annotations": [row["annotation"] for row in grouped]}

        with docs_path.open(encoding="utf-8") as docs_file, content_path.open(
            encoding="utf-8"
        ) as contents_file:
            for doc_line, content_line in zip(docs_file, contents_file, strict=True):
                doc = json.loads(doc_line)
                content_doc = json.loads(content_line)
                formats = {item["tool_name"]: item for item in content_doc["contents"]}
                if self._config.ocr_engine not in formats:
                    logger.warning(
                        "No %s common format for %s; found %s. Skipping it.",
                        self._config.ocr_engine,
                        doc["name"],
                        list(formats),
                    )
                    continue
                common = formats[self._config.ocr_engine]["common_format"]
                tokens = list(common["tokens"])
                if not tokens:
                    logger.warning("No tokens in common format for %s. Skipping it", doc["name"])
                    continue
                pages = common["structures"]["pages"]
                page_ranges = pages["structure_value"]
                page_boxes = pages["positions"]
                token_boxes = common["positions"]
                for question, answers in _annotations_for_document(doc):
                    if not answers:
                        answers = ["None"]
                    unique_answers = list(set(answers))
                    spans = [_answer_span(tokens, answer) for answer in unique_answers]
                    page_idx = 0
                    if len(page_ranges) > 1 and spans and spans[0][0] > -1:
                        for idx, bounds in enumerate(page_ranges):
                            if bounds[0] <= spans[0][0] <= bounds[1]:
                                page_idx = idx
                                break
                    start, end = page_ranges[page_idx]
                    page_box = page_boxes[page_idx]
                    width, height = page_box[2], page_box[3]
                    local_boxes = [
                        [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
                        for box in token_boxes[start : end + 1]
                    ]
                    current = {
                        "sample_id": doc["name"],
                        "page_idx": page_idx,
                        "page_size": (width, height),
                        "tokens": tokens[start : end + 1],
                        "bboxes": local_boxes,
                        "annotation": {
                            "question": question,
                            "answers": unique_answers,
                            "answer_starts": [
                                s - start if s >= 0 else -1 for s, _, _ in spans
                            ],
                            "answer_ends": [
                                e - start if e >= 0 else -1 for _, e, _ in spans
                            ],
                        },
                    }
                    if grouped and (
                        grouped[-1]["sample_id"] != current["sample_id"]
                        or grouped[-1]["page_idx"] != current["page_idx"]
                    ):
                        flushed = flush_group()
                        if flushed is not None:
                            yield flushed
                        grouped.clear()
                    grouped.append(current)
        flushed = flush_group()
        if flushed is not None:
            yield flushed

class DueBenchmarkExtractiveQA(Dataset[SinglePageDocumentInstance, DueBenchmarkConfig]):
    """DUE's original QA/PDF source with DocDjinn's default OCR choice."""

    __module_name__ = "due_benchmark"

    def _download_urls(self) -> list[UrlSpec]:
        name = self.config.config_name[2:]
        return [
            UrlSpec(
                url=url.format(name=name),
                url_ext=".tar.gz",
                rel_output_file_path=f"{key}/{name}.tar.gz",
            )
            for key, url in _DATA_URLS.items()
        ]

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="Extractive question answering datasets from the DUE benchmark.",
            homepage="https://duebenchmark.com/",
            dataset_labels=DatasetLabels(),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLITS)

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> _DueSplit:
        return _DueSplit(Path(data_dir), self.config, split)

    def _build_input_transform(
        self,
    ) -> Callable[[dict[str, Any]], SinglePageDocumentInstance]:
        def transform(sample: dict[str, Any]) -> SinglePageDocumentInstance:
            pages = convert_from_path(
                sample["pdf_path"],
                first_page=sample["page_idx"] + 1,
                last_page=sample["page_idx"] + 1,
                dpi=self.config.image_dpi,
                size=sample["page_size"],
            )
            if not pages:
                raise ValueError(
                    f"Could not load page {sample['page_idx']} from {sample['pdf_path']}"
                )
            elements = ElementArray.from_words(
                texts=sample["tokens"],
                bboxes=sample["bboxes"],
                bbox_mode=BoundingBoxMode.XYXY,
                normalized=True,
            )
            instance = SinglePageDocumentInstance(
                sample_id=f"{sample['sample_id']}-{sample['page_idx']}",
                visual=Image(content=pages[0]),
                content=DocumentContent(elements=elements),
            )
            qa_pairs = [
                QAPair(
                    id=index,
                    question_text=annotation["question"],
                    answer_text=annotation["answers"][0],
                    start=annotation["answer_starts"][0]
                    if annotation["answer_starts"]
                    else None,
                    end=annotation["answer_ends"][0]
                    if annotation["answer_ends"]
                    else None,
                    alternative_answers=list(annotation["answers"]),
                )
                for index, annotation in enumerate(sample["annotations"])
            ]
            return instance.add_annotation(QuestionAnsweringAnnotation(qa_pairs=qa_pairs))

        return transform
