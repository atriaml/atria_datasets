from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import DatasetSplitType, SinglePageDocumentInstance
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image


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


def fhswf_german_handwriting(
    config_name: str = "default",
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
) -> FHSWFGermanHandwriting:
    """Build the FHSWF German handwriting dataset."""
    return FHSWFGermanHandwriting(
        config=HuggingfaceDatasetConfig(config_name=config_name),
        dataset_dir_name="fhswf_german_handwriting",
        data_dir=data_dir,
        access_token=access_token,
        split=split,
    )
