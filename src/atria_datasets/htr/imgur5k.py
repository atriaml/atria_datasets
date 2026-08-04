from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import requests
import tqdm
from atria_core.datasets import Dataset, DatasetConfig
from atria_core.datasets._download._download_manager import (
    AtriaDownloadManager,
    UrlSpec,
)
from atria_core.logger import get_logger
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    SinglePageDocumentInstance,
)
from atria_core.types._generic._annotations import OCRAnnotation
from atria_core.types._generic._image import Image
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atria_datasets.htr._common import get_image_size
from atria_datasets.registry import datasets

_HOMEPAGE = "https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset"
_UPSTREAM_REVISION = "756a9ac9ed5201345661e1d9b7a5eb53502b97d5"
_UPSTREAM_RAW = (
    "https://raw.githubusercontent.com/facebookresearch/"
    f"IMGUR5K-Handwriting-Dataset/{_UPSTREAM_REVISION}/dataset_info"
)
_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "val",
    DatasetSplitType.test: "test",
}
_INFO_FILENAMES = [
    "imgur5k_hashes.lst",
    "imgur5k_annotations_train.json",
    "imgur5k_annotations_val.json",
    "imgur5k_annotations_test.json",
]
_DOWNLOAD_WORKERS = 8
_DOWNLOAD_TIMEOUT_SECONDS = 30

logger = get_logger(__name__)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_hashes(path: Path) -> dict[str, str]:
    return {
        image_id: checksum
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for image_id, checksum in [line.split()]
    }


def _annotation_image_ids(info_dir: Path) -> set[str]:
    image_ids: set[str] = set()
    for split_name in _SPLIT_NAMES.values():
        path = info_dir / f"imgur5k_annotations_{split_name}.json"
        image_ids.update(json.loads(path.read_text(encoding="utf-8"))["index_id"])
    return image_ids


def _parse_bounding_box(value: str | list[float]) -> tuple[float, ...] | None:
    if value == ".":
        return None
    if isinstance(value, str):
        values = value.strip().strip("[]()").split(",")
    else:
        values = value
    coordinates = tuple(float(item) for item in values)
    if len(coordinates) != 5:
        raise ValueError(f"Expected five IMGUR5K bounding-box values, got {value!r}")
    return coordinates


def _download_image(image_id: str, checksum: str, image_dir: Path) -> bool:
    image_path = image_dir / f"{image_id}.jpg"
    if image_path.exists():
        if _md5(image_path) == checksum:
            return True
        image_path.unlink()

    incomplete_path = image_path.with_suffix(".jpg.incomplete")
    incomplete_path.unlink(missing_ok=True)
    try:
        with requests.get(
            f"https://i.imgur.com/{image_id}.jpg",
            headers={"User-Agent": "Atria IMGUR5K downloader"},
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            content = response.content
        if len(content) < 100:
            return False
        digest = hashlib.md5(content, usedforsecurity=False).hexdigest()
        if digest != checksum:
            return False
        incomplete_path.write_bytes(content)
        incomplete_path.replace(image_path)
        return True
    except (OSError, requests.RequestException):
        incomplete_path.unlink(missing_ok=True)
        return False


def _download_images(info_dir: Path, image_dir: Path) -> tuple[int, int]:
    hashes = _parse_hashes(info_dir / "imgur5k_hashes.lst")
    image_ids = sorted(_annotation_image_ids(info_dir))
    missing_hashes = [image_id for image_id in image_ids if image_id not in hashes]
    if missing_hashes:
        logger.warning(
            "Missing official hashes for %d IMGUR5K images", len(missing_hashes)
        )
    download_items = [
        (image_id, hashes[image_id]) for image_id in image_ids if image_id in hashes
    ]

    def download(item: tuple[str, str]) -> bool:
        return _download_image(*item, image_dir)

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(download, download_items),
                total=len(download_items),
                desc="Downloading IMGUR5K images",
                unit="image",
            )
        )
    return sum(results), len(download_items)


@datasets.register("imgur5k")
@pydantic_dataclass(frozen=True)
class IMGUR5KConfig(DatasetConfig):
    def build_module(self, **kwargs: Any) -> IMGUR5K:
        return IMGUR5K(self, **kwargs)


class IMGUR5K(Dataset[IMGUR5KConfig, SinglePageDocumentInstance]):
    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        data_root = Path(data_dir)
        root = data_root / "imgur5k"
        info_dir = root / "dataset_info"
        image_dir = root / "images"
        info_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        annotation_paths = [
            info_dir / f"imgur5k_annotations_{split_name}.json"
            for split_name in _SPLIT_NAMES.values()
        ]
        hashes_path = info_dir / "imgur5k_hashes.lst"

        # Preserve upstream-script layouts that predate automatic downloading.
        if all(path.exists() for path in annotation_paths) and not hashes_path.exists():
            return {"imgur5k": root}

        manager = AtriaDownloadManager(
            data_dir=data_root, download_dir=data_root / ".download_cache"
        )
        manager.download_and_extract(
            [
                UrlSpec(
                    url=f"{_UPSTREAM_RAW}/{filename}",
                    url_ext=Path(filename).suffix,
                    rel_output_file_path=f"imgur5k/dataset_info/{filename}",
                )
                for filename in _INFO_FILENAMES
            ],
            extract=False,
        )

        complete_marker = root / ".images_download_complete"
        if not complete_marker.exists():
            valid, attempted = _download_images(info_dir, image_dir)
            complete_marker.write_text(
                f"valid={valid}\nattempted={attempted}\n", encoding="utf-8"
            )
            logger.info("Downloaded %d/%d valid IMGUR5K images", valid, attempted)

        return {"imgur5k": root}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description="IMGUR5K in-the-wild English handwriting with rotated word boxes.",
            homepage=_HOMEPAGE,
            license="CC BY-NC 4.0",
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> list[tuple[Path, OCRAnnotation]]:
        root = Path(data_dir) / "imgur5k"
        info_path = (
            root / "dataset_info" / f"imgur5k_annotations_{_SPLIT_NAMES[split]}.json"
        )
        data = json.loads(info_path.read_text(encoding="utf-8"))
        samples: list[tuple[Path, OCRAnnotation]] = []
        for index, metadata in data["index_id"].items():
            image_path = root / "images" / f"{index}.jpg"
            if not image_path.exists():
                candidate = Path(metadata["image_path"])
                image_path = candidate if candidate.is_absolute() else root / candidate
            if not image_path.exists():
                continue
            width, height = get_image_size(image_path)
            texts, bboxes, angles = [], [], []
            for annotation_id in data["index_to_ann_map"][index]:
                item = data["ann_id"][annotation_id]
                bounding_box = _parse_bounding_box(item["bounding_box"])
                if bounding_box is None:
                    continue
                xc, yc, box_width, box_height, angle = bounding_box
                texts.append(item["word"])
                bboxes.append(
                    [
                        (xc - box_width / 2) / width,
                        (yc - box_height / 2) / height,
                        (xc + box_width / 2) / width,
                        (yc + box_height / 2) / height,
                    ]
                )
                angles.append(angle)
            if texts:
                annotation = replace(
                    OCRAnnotation.from_words(texts, np.clip(bboxes, 0.0, 1.0)),
                    angles=np.asarray(angles),
                )
                samples.append((image_path, annotation))
        return samples

    def _build_input_transform(self) -> Callable[[Any], SinglePageDocumentInstance]:
        def transform(sample: tuple[Path, OCRAnnotation]) -> SinglePageDocumentInstance:
            image_path, annotation = sample
            return SinglePageDocumentInstance(
                sample_id=image_path.stem, visual=Image(file_path=str(image_path))
            ).add_annotation(annotation)

        return transform
