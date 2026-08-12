from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import QAPair, QuestionAnsweringAnnotation, TextInstance
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets


@datasets.register(name="squad")
@pydantic_dataclass(frozen=True)
class SquadConfig(HuggingfaceDatasetConfig):
    def build_module(self, **kwargs: Any) -> Squad:
        return Squad(repo="squad", config=self, **kwargs)


class InputTransform:
    def __call__(self, sample: dict[str, Any]) -> TextInstance:
        answers = sample["answers"]
        qa_pair = QAPair(
            id=0,
            question_text=sample["question"],
            answer_text=answers["text"][0],
            start=answers["answer_start"][0] if answers["answer_start"] else None,
        )
        return TextInstance(
            sample_id=sample["id"], text=sample["context"]
        ).add_annotation(annotation=QuestionAnsweringAnnotation(qa_pairs=[qa_pair]))


class Squad(HuggingfaceDataset[SquadConfig, TextInstance]):
    def _build_input_transform(self) -> Callable[[Any], TextInstance]:
        return InputTransform()
