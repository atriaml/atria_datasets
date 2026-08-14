"""Parser for the ASCII metadata distributed with the IAM databases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IAMRecord:
    sample_id: str
    segmentation_status: str
    graylevel: int
    components: int
    x: int
    y: int
    width: int
    height: int
    text: str
    grammatical_tag: str | None = None

    @property
    def form_id(self) -> str:
        return "-".join(self.sample_id.split("-")[:2])

    @property
    def line_id(self) -> str:
        return "-".join(self.sample_id.split("-")[:3])


@dataclass(frozen=True)
class IAMFormRecord:
    form_id: str
    writer_id: str
    sentence_count: int
    segmentation_status: str
    line_count: int
    segmented_line_count: int
    word_count: int
    segmented_word_count: int


def parse_iam_forms(path: str | Path) -> dict[str, IAMFormRecord]:
    """Parse IAM ``forms.txt`` records keyed by form ID."""
    records: dict[str, IAMFormRecord] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 fields, got {len(fields)}"
                )
            form_id, writer_id, sentences, status, *counts = fields
            try:
                record = IAMFormRecord(
                    form_id=form_id,
                    writer_id=writer_id,
                    sentence_count=int(sentences),
                    segmentation_status=status,
                    line_count=int(counts[0]),
                    segmented_line_count=int(counts[1]),
                    word_count=int(counts[2]),
                    segmented_word_count=int(counts[3]),
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric metadata"
                ) from error
            if form_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {form_id!r}")
            records[form_id] = record
    return records


def parse_iam_ascii(path: str | Path) -> dict[str, IAMRecord]:
    """Parse IAM ``lines.txt`` or ``words.txt`` into records keyed by id."""
    records: dict[str, IAMRecord] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=8)
            if len(fields) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 fields, got {len(fields)}"
                )
            try:
                if fields[7].lstrip("-").isdigit():
                    (
                        sample_id,
                        status,
                        graylevel,
                        components,
                        x,
                        y,
                        width,
                        height,
                        text,
                    ) = fields
                    record = IAMRecord(
                        sample_id=sample_id,
                        segmentation_status=status,
                        graylevel=int(graylevel),
                        components=int(components),
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                        text=text.replace("|", " "),
                    )
                else:
                    sample_id, status, graylevel, x, y, width, height, tag, text = (
                        fields
                    )
                    record = IAMRecord(
                        sample_id=sample_id,
                        segmentation_status=status,
                        graylevel=int(graylevel),
                        components=0,
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                        text=text.replace("|", " "),
                        grammatical_tag=tag,
                    )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric metadata"
                ) from error
            if sample_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
            records[sample_id] = record
    return records


def parse_iam_split(path: str | Path) -> set[str]:
    """Read an Aachen split file containing one form id per line."""
    with Path(path).open(encoding="utf-8") as stream:
        return {
            line.strip()
            for line in stream
            if line.strip() and not line.lstrip().startswith("#")
        }
