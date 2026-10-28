"""DocLayNet document-category and layout-analysis dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from atria_core.datasets import HuggingfaceDataset
from atria_core.types import (
    ClassificationAnnotation,
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


class DocLayNet(HuggingfaceDataset[SinglePageDocumentInstance, LayoutDatasetConfig]):
    """DocLayNet with its original document and layout annotations."""

    __module_name__ = "doclaynet"
    __hf_repo__ = "ds4sd/DocLayNet"
    __hf_config_name__ = "2022.08"

    _DOC_CLASSES = [
        "financial_reports",
        "scientific_articles",
        "laws_and_regulations",
        "government_tenders",
        "manuals",
        "patents",
    ]
    _LAYOUT_CLASSES = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title ",
    ]

    def _build_split_iterator(self, split: DatasetSplitType, data_dir: str) -> Any:
        return limit_split(
            iterator=super()._build_split_iterator(split, data_dir),
            limit=sample_cap_for_split(config=self.config, split=split),
        )

    def _metadata(self) -> DatasetMetadata:
        return replace(
            super()._metadata(),
            dataset_labels=DatasetLabels(
                classification=self._DOC_CLASSES, layout=self._LAYOUT_CLASSES
            ),
        )

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        def transform(sample: dict[str, Any]) -> SinglePageDocumentInstance:
            image = Image(content=sample["image"])
            objects = annotated_objects(
                annotations=sample["objects"],
                class_names=self._LAYOUT_CLASSES,
                category_offset=0,
                width=image.width,
                height=image.height,
            )
            doc_category = sample["doc_category"]
            return (
                SinglePageDocumentInstance(
                    sample_id=f"sample_{sample['image_id']}", visual=image
                )
                .add_annotation(
                    ClassificationAnnotation(
                        label_value=self._DOC_CLASSES.index(doc_category),
                        label_name=doc_category,
                    )
                )
                .add_annotation(layout_annotation(objects))
            )

        return transform
