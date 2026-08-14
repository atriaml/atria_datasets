"""Shared PAGE-XML (PRImA/READ schema family) ground-truth reader.

Builds a page->block->line->word OCRAnnotation hierarchy from a single
PAGE-XML file, the same parallel-array construction pattern used by
`atria_core.extractors._tesseract.TesseractExtractor`, but reading
<TextRegion>/<TextLine>/<Word>/<Coords>/<TextEquiv> instead of
pytesseract's data dict. Word-level elements are optional -- most
historical PAGE-XML ground truth is line-level only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from atria_core.types._generic._annotations import OCRAnnotation
from atria_core.types._generic._elements import OCRLevel
from lxml import etree

_ROOT_PARENT = -1
_BARE_AMPERSAND = re.compile(rb"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)")


def _parse_xml(xml_path: Path) -> etree._Element:
    """Parse PAGE XML while preserving bare ampersands in legacy transcripts."""
    xml = xml_path.read_bytes()
    if _BARE_AMPERSAND.search(xml):
        xml = _BARE_AMPERSAND.sub(b"&amp;", xml)
    return etree.fromstring(text=xml)


def _parse_points(points: str) -> np.ndarray:
    """`"x1,y1 x2,y2 ..."` -> (P, 2) float array."""
    coords = [tuple(map(float, pair.split(","))) for pair in points.split()]
    return np.asarray(coords, dtype=np.float64)


def _coords(element: etree._Element, ns: dict[str, str]) -> np.ndarray | None:
    coords_el = element.find(path="pc:Coords", namespaces=ns)
    if coords_el is None or not coords_el.get("points"):
        return None
    return _parse_points(points=coords_el.get("points"))


def _text(element: etree._Element, ns: dict[str, str]) -> str:
    """First (or index="0") <TextEquiv><Unicode> under `element`."""
    text_equiv = element.find(path="pc:TextEquiv", namespaces=ns)
    if text_equiv is None:
        return ""
    unicode_el = text_equiv.find(path="pc:Unicode", namespaces=ns)
    return (unicode_el.text or "").strip() if unicode_el is not None else ""


def _reading_order(page: etree._Element, ns: dict[str, str]) -> list[str] | None:
    """Region ids from <ReadingOrder>/<OrderedGroup>, ascending by `index`,
    or None if the file has no reading-order element (caller falls back to
    document order)."""
    reading_order = page.find(path="pc:ReadingOrder", namespaces=ns)
    if reading_order is None:
        return None
    refs = reading_order.findall(path=".//pc:RegionRefIndexed", namespaces=ns)
    if not refs:
        return None
    ordered = sorted(refs, key=lambda r: int(r.get("index", 0)))
    return [r.get("regionRef") for r in ordered]


def parse_page_xml(
    xml_path: Path, image_size: tuple[float, float] | None = None
) -> OCRAnnotation:
    """Parses a PAGE-XML ground-truth file into a single OCRAnnotation
    holding the full page/block/line/word hierarchy. Coordinates come from
    each element's <Coords points="..."> and are normalized to [0, 1] using
    the page's pixel size -- `image_size` (width, height), if given,
    overrides the XML's own imageWidth/imageHeight attributes (which are
    sometimes stale relative to the actual on-disk image).

    The PAGE-XML namespace is read off the root element rather than
    hardcoded, since it varies by schema vintage (2009-2019) across
    READ-era Zenodo records; the element names used here (Page, TextRegion,
    TextLine, Word, Coords, TextEquiv, Unicode, ReadingOrder,
    RegionRefIndexed) are stable across all of them.
    """
    root = _parse_xml(xml_path=xml_path)
    ns_uri = etree.QName(root).namespace
    ns = {"pc": ns_uri} if ns_uri else {}

    page = (
        root.find(path="pc:Page", namespaces=ns) if ns_uri else root.find(path="Page")
    )
    if page is None:
        raise ValueError(f"No <Page> element found in {xml_path}")

    if image_size is not None:
        width, height = image_size
    else:
        raw_width, raw_height = page.get("imageWidth"), page.get("imageHeight")
        if raw_width is None or raw_height is None:
            raise ValueError(
                f"{xml_path}: <Page> has no imageWidth/imageHeight and no "
                "image_size was given"
            )
        width, height = float(raw_width), float(raw_height)

    ids: list[int] = []
    parent_ids: list[int] = []
    levels: list[int] = []
    bboxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    polygons: list[np.ndarray | None] = []

    def _add(
        level: OCRLevel, parent_id: int, polygon_px: np.ndarray | None, text: str
    ) -> int:
        element_id = len(ids)
        norm_bbox: tuple[float, float, float, float]
        if polygon_px is not None:
            xs, ys = polygon_px[:, 0], polygon_px[:, 1]
            clipped_bbox = np.clip(
                (
                    xs.min() / width,
                    ys.min() / height,
                    xs.max() / width,
                    ys.max() / height,
                ),
                0.0,
                1.0,
            )
            norm_bbox = (
                float(clipped_bbox[0]),
                float(clipped_bbox[1]),
                float(clipped_bbox[2]),
                float(clipped_bbox[3]),
            )
            norm_polygon = np.clip(polygon_px / np.array([width, height]), 0.0, 1.0)
        else:
            norm_bbox = (0.0, 0.0, 1.0, 1.0)
            norm_polygon = None

        ids.append(element_id)
        parent_ids.append(parent_id)
        levels.append(level.value)
        bboxes.append(norm_bbox)
        texts.append(text)
        polygons.append(norm_polygon)
        return element_id

    page_id = _add(
        level=OCRLevel.page, parent_id=_ROOT_PARENT, polygon_px=None, text=""
    )

    regions = {
        r.get("id"): r for r in page.findall(path="pc:TextRegion", namespaces=ns)
    }
    ordered_ids = _reading_order(page=page, ns=ns)
    if ordered_ids:
        ordered_regions = [regions[rid] for rid in ordered_ids if rid in regions]
        ordered_regions += [
            region for rid, region in regions.items() if rid not in ordered_ids
        ]
    else:
        ordered_regions = list(regions.values())

    for region in ordered_regions:
        region_id = _add(
            level=OCRLevel.block,
            parent_id=page_id,
            polygon_px=_coords(element=region, ns=ns),
            text=_text(element=region, ns=ns),
        )
        for line in region.findall(path="pc:TextLine", namespaces=ns):
            line_id = _add(
                level=OCRLevel.line,
                parent_id=region_id,
                polygon_px=_coords(element=line, ns=ns),
                text=_text(element=line, ns=ns),
            )
            for word in line.findall(path="pc:Word", namespaces=ns):
                _add(
                    level=OCRLevel.word,
                    parent_id=line_id,
                    polygon_px=_coords(element=word, ns=ns),
                    text=_text(element=word, ns=ns),
                )

    # Prefer aligned line text. Weakly annotated PAGE releases may instead
    # store one transcript on each region or directly on the page.
    line_text = "\n".join(
        text
        for level, text in zip(levels, texts, strict=True)
        if level == OCRLevel.line.value and text
    )
    region_text = "\n".join(
        text
        for level, text in zip(levels, texts, strict=True)
        if level == OCRLevel.block.value and text
    )
    texts[page_id] = line_text or region_text or _text(element=page, ns=ns)

    lengths = np.array([0 if p is None else len(p) for p in polygons])
    p_max = int(lengths.max()) if len(lengths) else 0
    segmentations = np.full((len(ids), p_max, 2), np.nan)
    for i, polygon in enumerate(polygons):
        if polygon is not None:
            segmentations[i, : len(polygon)] = polygon

    annotation = OCRAnnotation(
        ids=np.array(ids),
        parent_ids=np.array(parent_ids),
        levels=np.array(levels),
        bboxes=np.asarray(bboxes, dtype=np.float64),
        texts=np.asarray(texts, dtype=object),
        segmentations=segmentations,
        segmentation_lengths=lengths,
    )
    annotation.validate_hierarchy()
    return annotation
