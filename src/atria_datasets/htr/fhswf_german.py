from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import (
    DatasetSplitType,
    Image,
    OCRLevel,
    SinglePageDocumentInstance,
    TranscriptionAnnotation,
)


class InputTransform:
    def __call__(
        self, sample: tuple[int, dict[str, Any]]
    ) -> SinglePageDocumentInstance:
        sample_idx, sample_dict = sample
        return SinglePageDocumentInstance(
            sample_id=f"Sample_{sample_idx}", visual=Image(content=sample_dict["image"])
        ).add_annotation(
            annotation=TranscriptionAnnotation(
                text=sample_dict["text"], level=OCRLevel.line
            )
        )


class FHSWFGermanHandwriting(
    HuggingfaceDataset[SinglePageDocumentInstance, HuggingfaceDatasetConfig]
):
    """FHSWF German handwritten line images and transcriptions."""

    __module_name__ = "fhswf_german_handwriting"

    def __init__(
        self, *, config: HuggingfaceDatasetConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(repo="fhswf/german_handwriting", config=config, **kwargs)

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        return enumerate(
            self._builder._as_streaming_dataset_single(self._hf_split_generators[split])
        )
