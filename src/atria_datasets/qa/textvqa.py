from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import (
    QAPair,
    QuestionAnsweringAnnotation,
    SinglePageDocumentInstance,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets

_IMAGES_SUBDIR = "images"


@datasets.register(name="textvqa")
@pydantic_dataclass(frozen=True)
class TextVqaConfig(HuggingfaceDatasetConfig):
    def build_module(self, **kwargs: Any) -> TextVqa:
        return TextVqa(repo="lmms-lab/textvqa", config=self, **kwargs)


class InputTransform:
    def __init__(self, images_dir: Path) -> None:
        self._images_dir = images_dir
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, sample: dict[str, Any]) -> SinglePageDocumentInstance:
        answers: list[str] = sample["answers"]
        qa_pair = QAPair(
            id=0,
            question_text=sample["question"],
            answer_text=answers[0],
            alternative_answers=list(answers),
        )
        sample_id = str(sample["question_id"])
        image_path = self._images_dir / f"{sample_id}.jpg"
        if not image_path.exists():
            sample["image"].convert("RGB").save(image_path, format="JPEG")
        return SinglePageDocumentInstance.from_image(
            image_path, sample_id=sample_id
        ).add_annotation(annotation=QuestionAnsweringAnnotation(qa_pairs=[qa_pair]))


class TextVqa(HuggingfaceDataset[TextVqaConfig, SinglePageDocumentInstance]):
    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform(self.data_dir / _IMAGES_SUBDIR)
