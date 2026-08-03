from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from atria_core.datasets._cacher import Cacher, FileStorageType
from atria_core.datasets._hf_dataset import HuggingfaceDataset, HuggingfaceDatasetConfig
from atria_core.types import SinglePageDocumentInstance
from atria_core.types._generic._annotations import TranscriptionAnnotation
from atria_core.types._generic._image import Image
from atria_core.visualizers import visualize
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.registry import datasets


@datasets.register("fhswf_german_handwriting")
@pydantic_dataclass(frozen=True)
class FHSWFGermanHandwritingConfig(HuggingfaceDatasetConfig):
    config_name: str = "default"

    def build_module(self) -> FHSWFGermanHandwriting:
        return FHSWFGermanHandwriting("fhswf/german_handwriting", config=self)


class InputTransform:
    def __call__(self, sample: dict[str, Any]) -> SinglePageDocumentInstance:
        return SinglePageDocumentInstance(
            sample_id=str(uuid.uuid4()), visual=Image(content=sample["image"])
        ).add_annotation(TranscriptionAnnotation(text=sample["text"]))


class FHSWFGermanHandwriting(
    HuggingfaceDataset[FHSWFGermanHandwritingConfig, SinglePageDocumentInstance]
):
    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        return InputTransform()


dataset = FHSWFGermanHandwritingConfig().build_module()
cached = Cacher(FileStorageType.MSGPACK).cache(dataset)
for sample in cached.train:
    sample = sample.load()
    visualize(sample, output_dir="./test")
