from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import DatasetSplitType, SinglePageDocumentInstance
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import dataset_configs


@dataset_configs.register(name="fhswf_german_handwriting")
@pydantic_dataclass(frozen=True)
class FHSWFGermanHandwritingConfig(HuggingfaceDatasetConfig):
    config_name: str = "default"

    def build_module(self, **kwargs: Any) -> FHSWFGermanHandwriting:
        return FHSWFGermanHandwriting(
            repo="fhswf/german_handwriting", config=self, **kwargs
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
    HuggingfaceDataset[FHSWFGermanHandwritingConfig, SinglePageDocumentInstance]
):
    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        return enumerate(
            self._builder._as_streaming_dataset_single(self._hf_split_generators[split])
        )
