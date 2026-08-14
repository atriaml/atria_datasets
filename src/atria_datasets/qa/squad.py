from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset
from atria_core.types import (
    DatasetSplitType,
    QAPair,
    QuestionAnsweringAnnotation,
    TextInstance,
)


class InputTransform:
    """Turns a raw SQuAD record into a TextInstance carrying its question-answer pair."""

    def __call__(self, sample: dict[str, Any]) -> TextInstance:
        answers = sample["answers"]
        qa_pair = QAPair(
            id=0,
            question_text=sample["question"],
            answer_text=answers["text"][0],
            start=answers["answer_start"][0] if answers["answer_start"] else None,
            alternative_answers=list(answers["text"]),
        )
        return TextInstance(
            sample_id=sample["id"], text=sample["context"]
        ).add_annotation(annotation=QuestionAnsweringAnnotation(qa_pairs=[qa_pair]))


class Squad(HuggingfaceDataset[TextInstance]):
    """SQuAD: questions posed against Wikipedia passages, with answer spans."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(repo="squad", **kwargs)

    def _build_input_transform(self) -> Callable[[Any], TextInstance]:
        return InputTransform()


def squad(
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> Squad:
    """Build the SQuAD reading-comprehension dataset."""
    return Squad(
        dataset_dir_name="squad",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
