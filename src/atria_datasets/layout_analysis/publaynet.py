"""PubLayNet layout-analysis dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from atria_core.datasets import HuggingfaceDataset
from atria_core.types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    Image,
    SinglePageDocumentInstance,
)
from atria_datasets.layout_analysis.config import LayoutDatasetConfig
from atria_datasets.layout_analysis.utilities import (
    annotated_objects,
    layout_annotation,
    limit_split,
    sample_cap_for_split,
)


class PubLayNet(HuggingfaceDataset[SinglePageDocumentInstance, LayoutDatasetConfig]):
    """PubLayNet with its original layout labels and optional split caps."""

    __module_name__ = "publaynet"
    __hf_repo__ = "jordanparker6/publaynet"
    __hf_config_name__ = "default"
    _CLASSES = ["text", "title", "list", "table", "figure"]

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        return limit_split(
            iterator=super()._build_split_iterator(split, data_dir),
            limit=sample_cap_for_split(config=self.config, split=split),
        )

    def _metadata(self) -> DatasetMetadata:
        return replace(
            super()._metadata(), dataset_labels=DatasetLabels(layout=self._CLASSES)
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        def transform(sample: dict[str, Any]) -> SinglePageDocumentInstance:
            image = Image(content=sample["image"])
            objects = annotated_objects(
                annotations=sample["annotations"],
                class_names=self._CLASSES,
                category_offset=1,
                width=image.width,
                height=image.height,
            )
            return SinglePageDocumentInstance(
                sample_id=str(sample["id"]), visual=image
            ).add_annotation(layout_annotation(objects))

        return transform
