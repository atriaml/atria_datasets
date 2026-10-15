"""Port of the original Atria Tesseract hOCR parser."""

from __future__ import annotations

import ast

import numpy as np
from bs4 import BeautifulSoup

from atria_core.types import DocumentContent, OCRAnnotation, OCRLevel


def parse_tesseract_hocr(raw_hocr: str) -> DocumentContent:
    """Apply the original OCR loading and XML parsing behavior."""
    # The original Atria OCR loader handled OCR archives containing Python's
    # string representation of bytes (e.g. ``b'<?xml ...'``) before parsing.
    if raw_hocr.startswith("b'"):
        raw_hocr = ast.literal_eval(raw_hocr).decode("utf-8")

    soup = BeautifulSoup(raw_hocr, features="xml")

    pages = soup.findAll("div", {"class": "ocr_page"})
    image_size_str = pages[0]["title"].split("; bbox")[1]
    width, height = map(int, image_size_str[4 : image_size_str.find(";")].split())

    words: list[str] = []
    boxes: list[list[float]] = []
    angles: list[float] = []
    confidences: list[float] = []
    ocr_words = soup.findAll("span", {"class": "ocrx_word"})
    for word in ocr_words:
        title = word["title"]
        confidence = float(title[title.find(";") + 10 :])
        if word.text.strip() == "":
            continue

        text_angle = 0.0
        parent_title = word.parent["title"]
        if "textangle" in parent_title:
            text_angle = float(parent_title.split("textangle")[1][1:3])

        x0, y0, x1, y1 = map(int, title[5 : title.find(";")].split())
        words.append(word.text.strip())
        boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
        angles.append(text_angle)
        confidences.append(confidence)

    count = len(words)
    elements = OCRAnnotation(
        ids=np.arange(count, dtype=np.int64),
        parent_ids=np.full(count, -1, dtype=np.int64),
        levels=np.full(count, OCRLevel.word.value, dtype=np.int64),
        bboxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        normalized=True,
        texts=np.asarray(words, dtype=object),
        angles=np.asarray(angles, dtype=np.float64),
        confs=np.asarray(confidences, dtype=np.float64),
    )
    return DocumentContent(elements=elements)
