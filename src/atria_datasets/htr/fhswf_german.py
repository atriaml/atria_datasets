from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import SinglePageDocumentInstance
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets


@datasets.register(name="fhswf_german_handwriting")
@pydantic_dataclass(frozen=True)
class FHSWFGermanHandwritingConfig(HuggingfaceDatasetConfig):
    config_name: str = "default"

    def build_module(self) -> FHSWFGermanHandwriting:
        return FHSWFGermanHandwriting(repo="fhswf/german_handwriting", config=self)


class InputTransform:
    def __call__(self, sample: dict[str, Any]) -> SinglePageDocumentInstance:
        sample_id = next(
            (
                str(sample[key]).strip()
                for key in ("sample_id", "id", "image_id", "file_name", "filename", "name")
                if key in sample and str(sample[key]).strip()
            ),
            None,
        )
        if sample_id is None:
            image = sample["image"]
            sample_id = getattr(image, "filename", "") or getattr(image, "path", "") or "sample"
            sample_id = str(sample_id).strip()
        if "/" in sample_id or "\\" in sample_id:
            sample_id = sample_id.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if "." in sample_id:
            sample_id = sample_id.rsplit(".", 1)[0]
        return SinglePageDocumentInstance(
            sample_id=sample_id, visual=Image(content=sample["image"])
        ).add_annotation(
            annotation=TranscriptionAnnotation(text=sample["text"], level=OCRLevel.line)
        )


class FHSWFGermanHandwriting(
    HuggingfaceDataset[FHSWFGermanHandwritingConfig, SinglePageDocumentInstance]
):
    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()
