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

    @property
    def form_id(self) -> str:
        return "-".join(self.sample_id.split("-")[:2])

    @property
    def line_id(self) -> str:
        return "-".join(self.sample_id.split("-")[:3])


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
            sample_id, status, graylevel, components, x, y, width, height, text = fields
            try:
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
