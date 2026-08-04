"""Shared iterators for HTR datasets with conventional sidecars."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

from atria_core.types import SinglePageDocumentInstance
from atria_core.types._generic._elements import OCRLevel
from atria_core.types._generic._image import Image
from lxml import etree
from PIL import Image as PILImage

from atria_datasets.parsers import parse_page_xml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _path_sample_id(path: Path) -> str:
    return path.stem


def _page_image_name(xml_path: Path) -> str | None:
    """Read PAGE's imageFilename without parsing the complete annotation."""
    for _, element in etree.iterparse(str(xml_path), events=("start",)):
        if etree.QName(element).localname == "Page":
            return element.get("imageFilename")
    return None


def _split_matches(path: Path, aliases: tuple[str, ...]) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return not aliases or any(alias.lower() in lowered for alias in aliases)


class PageXMLIterator(Sequence[tuple[Path, Path]]):
    """Discover PAGE-XML/image pairs independent of archive wrapper depth."""

    def __init__(
        self, root: str | Path, *, split_aliases: tuple[str, ...] = ()
    ) -> None:
        root = Path(root)
        images: dict[str, list[Path]] = {}
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.setdefault(path.name.lower(), []).append(path)
                images.setdefault(path.stem.lower(), []).append(path)

        self.samples: list[tuple[Path, Path]] = []
        paired_images: set[Path] = set()
        for xml_path in sorted(root.rglob("*.xml")):
            if not _split_matches(xml_path, split_aliases):
                continue
            image_name = _page_image_name(xml_path)
            keys = []
            if image_name:
                keys.extend(
                    [Path(image_name).name.lower(), Path(image_name).stem.lower()]
                )
            keys.append(xml_path.stem.lower())
            candidates = next((images[key] for key in keys if key in images), [])
            if candidates:
                # Prefer a nearby image when separate archives contain duplicate names.
                image_path = max(
                    candidates,
                    key=lambda item: len(
                        set(item.parent.parts) & set(xml_path.parent.parts)
                    ),
                )
                # Some releases duplicate identical PAGE files both beside
                # the image and in a page/ directory. A page image is one
                # sample, irrespective of how many annotation copies ship.
                resolved_image = image_path.resolve()
                if resolved_image in paired_images:
                    continue
                paired_images.add(resolved_image)
                self.samples.append((image_path, xml_path))

    @overload
    def __getitem__(self, index: int) -> tuple[Path, Path]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, Path]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class PageXMLTransform:
    def __call__(self, sample: tuple[Path, Path]) -> SinglePageDocumentInstance:
        image_path, xml_path = sample
        with PILImage.open(image_path) as image:
            annotation = parse_page_xml(xml_path, image_size=image.size)
        return SinglePageDocumentInstance(
            sample_id=_path_sample_id(image_path), visual=Image(file_path=str(image_path))
        ).add_annotation(annotation)


class TextSidecarIterator(Sequence[tuple[Path, str]]):
    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        images = {
            path.stem.lower(): path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        self.samples = [
            (
                images[text_path.stem.lower()],
                text_path.read_text(encoding="utf-8").strip(),
            )
            for text_path in sorted(root.rglob("*.txt"))
            if text_path.stem.lower() in images
        ]

    @overload
    def __getitem__(self, index: int) -> tuple[Path, str]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[Path, str]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class TextSidecarTransform:
    def __init__(self, *, level: OCRLevel | None = None) -> None:
        self.level = level

    def __call__(self, sample: tuple[Path, str]) -> SinglePageDocumentInstance:
        from atria_core.types._generic._annotations import TranscriptionAnnotation

        image_path, text = sample
        return SinglePageDocumentInstance(
            sample_id=_path_sample_id(image_path), visual=Image(file_path=str(image_path))
        ).add_annotation(TranscriptionAnnotation(text=text, level=self.level))
