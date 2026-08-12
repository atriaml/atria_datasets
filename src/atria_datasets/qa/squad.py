from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import (
    DocumentContent,
    Image,
    QAPair,
    QuestionAnsweringAnnotation,
    SinglePageDocumentInstance,
)
from PIL import Image as PILImage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets

#: SinglePageDocumentInstance.visual is a required field, but SQuAD is
#: pure text -- no page image ever gets read, so every sample shares one
#: trivial placeholder instead of loading real image bytes per sample.
_PLACEHOLDER_VISUAL = Image(content=PILImage.new("RGB", (1, 1)))


@datasets.register("squad")
@pydantic_dataclass(frozen=True)
class SquadConfig(HuggingfaceDatasetConfig):
    """SQuAD 1.1 (reading-comprehension QA): context + question +
    extractive answer. Streamed from the Hub, not bulk-downloaded."""

    config_name: str | None = None

    def build_module(self, **kwargs: Any) -> Squad:
        return Squad("squad", config=self, **kwargs)


class InputTransform:
    def __call__(self, sample: dict[str, Any]) -> SinglePageDocumentInstance:
        answers = sample["answers"]
        qa_pair = QAPair(
            id=0,
            question_text=sample["question"],
            answer_text=answers["text"][0],
            start=answers["answer_start"][0] if answers["answer_start"] else None,
        )
        return SinglePageDocumentInstance(
            sample_id=sample["id"],
            visual=_PLACEHOLDER_VISUAL,
            content=DocumentContent(_text=sample["context"]),
        ).add_annotation(QuestionAnsweringAnnotation(qa_pairs=[qa_pair]))


class Squad(HuggingfaceDataset[SquadConfig, SinglePageDocumentInstance]):
    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()
