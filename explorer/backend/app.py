from __future__ import annotations

import inspect
import io
import json
import queue
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from atria_core.datasets._cached_dataset import CachedDataset
from atria_core.datasets._cacher import Cacher
from atria_core.datasets._common import FileStorageType
from atria_core.datasets._constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    DEFAULT_ATRIA_CACHE_DIR,
)
from atria_core.datasets._snapshot import CACHED_DATASET_SNAPSHOT_KIND, DatasetSnapshot
from atria_core.datasets._snapshot_store import DatasetSnapshotStore
from atria_core.types._generic._annotations import AnnotationType, OCRAnnotation
from atria_core.types._generic._elements import OCRLevel
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import ImageDraw
from pydantic import BaseModel, Field

import atria_datasets

_RUNTIME_FACTORY_FIELDS = {"access_token", "data_dir", "split"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _factory_fields(factory: Any) -> list[dict[str, Any]]:
    fields = []
    for item in inspect.signature(factory).parameters.values():
        if item.kind in {item.VAR_POSITIONAL, item.VAR_KEYWORD}:
            continue
        if item.name in _RUNTIME_FACTORY_FIELDS:
            continue
        required = item.default is inspect.Parameter.empty
        default = None if required else _json_value(item.default)
        fields.append(
            {
                "name": item.name,
                "type": str(item.annotation),
                "required": required,
                "default": default,
            }
        )
    return fields


def _snapshot_dataset_id(snapshot: DatasetSnapshot) -> str:
    config_key = (
        snapshot.config_hash or snapshot.config_name or snapshot.dataset_class_name
    )
    return f"{snapshot.dataset_class_name}::{config_key}"


def _default_base_dir() -> Path:
    return Path(_DEFAULT_ATRIA_DATASETS_CACHE_DIR).expanduser()


def _normalize_base_dir(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


class SettingsUpdateRequest(BaseModel):
    base_dir: str | None = None


class PrepareJobRequest(BaseModel):
    name: str
    source_dir: str | None = None
    output_root: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ExplorerStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    source_dir TEXT,
                    output_root TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result_path TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                UPDATE jobs
                SET status = 'interrupted',
                    finished_at = COALESCE(finished_at, ?),
                    error = COALESCE(error, 'Explorer stopped before the job finished.')
                WHERE status IN ('queued', 'running')
                """,
                (_utc_now(),),
            )

    def get_setting(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_setting(self, key: str, value: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def create_job(
        self,
        request: PrepareJobRequest,
        *,
        normalized_source_dir: str | None,
        resolved_output_root: str,
    ) -> str:
        job_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, status, dataset_name, source_dir, output_root, config_json,
                    created_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    request.name,
                    normalized_source_dir,
                    resolved_output_root,
                    json.dumps(request.config, sort_keys=True),
                    _utc_now(),
                ),
            )
        return job_id

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM jobs
                ORDER BY datetime(created_at) DESC, id DESC
                """
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return None if row is None else self._job_row(row)

    def mark_running(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    started_at = ?,
                    finished_at = NULL,
                    error = NULL
                WHERE id = ?
                """,
                (_utc_now(), job_id),
            )

    def mark_complete(self, job_id: str, result_path: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    finished_at = ?,
                    result_path = ?,
                    error = NULL
                WHERE id = ?
                """,
                (_utc_now(), result_path, job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    finished_at = ?,
                    error = ?
                WHERE id = ?
                """,
                (_utc_now(), error, job_id),
            )

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "status": str(row["status"]),
            "dataset_name": str(row["dataset_name"]),
            "source_dir": row["source_dir"],
            "output_root": row["output_root"],
            "config": json.loads(str(row["config_json"])),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "result_path": row["result_path"],
            "error": row["error"],
        }


class ExplorerState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.store = ExplorerStore(
            Path(__file__).resolve().parent.parent / "state" / "explorer.db"
        )
        self.inventory: dict[str, Any] | None = None
        self.active_dataset_id: str | None = None
        self.active_dataset: CachedDataset[Any] | None = None
        self._job_queue: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._job_loop, daemon=True)
        self._worker.start()

    def configured_base_dir(self) -> str | None:
        return _normalize_base_dir(self.store.get_setting("base_dir"))

    def resolved_base_dir(self) -> Path:
        configured = self.configured_base_dir()
        if configured is None:
            return _default_base_dir()
        return Path(configured).expanduser()

    def set_base_dir(self, base_dir: str | None) -> dict[str, Any]:
        self.store.set_setting("base_dir", _normalize_base_dir(base_dir))
        return self.scan_inventory(force=True)

    def scan_inventory(self, *, force: bool = False) -> dict[str, Any]:
        with self.lock:
            if self.inventory is not None and not force:
                return self.inventory

            base_dir = self.resolved_base_dir()
            discovered: list[dict[str, Any]] = []
            invalid: list[dict[str, str]] = []

            if base_dir.exists():
                for snapshot in DatasetSnapshotStore.discover(
                    base_dir, snapshot_kind=CACHED_DATASET_SNAPSHOT_KIND
                ):
                    if snapshot.path is None:
                        invalid.append(
                            {
                                "path": str(base_dir.resolve()),
                                "reason": "Discovered snapshot is missing its path.",
                            }
                        )
                        continue

                    discovered.append(
                        self._prepared_dataset_payload(snapshot, base_dir)
                    )

            discovered.sort(
                key=lambda item: (
                    item["dataset_class_name"].lower(),
                    str(item["config_name"] or "").lower(),
                    item["path"],
                )
            )

            self.inventory = {
                "base_dir": self.configured_base_dir(),
                "resolved_base_dir": str(base_dir.resolve()),
                "default_base_dir": str(_default_base_dir().resolve()),
                "default_cache_root": str(Path(DEFAULT_ATRIA_CACHE_DIR).expanduser()),
                "datasets": discovered,
                "invalid_snapshots": invalid,
                "scanned_at": _utc_now(),
            }
            return self.inventory

    def _require_snapshot(self, dataset_id: str) -> DatasetSnapshot:
        inventory = self.scan_inventory()
        match = next(
            (item for item in inventory["datasets"] if item["id"] == dataset_id), None
        )
        if match is None:
            raise HTTPException(status_code=404, detail="Prepared dataset not found.")
        return DatasetSnapshot.load(match["path"])

    def _prepared_dataset_payload(
        self, snapshot: DatasetSnapshot, base_dir: Path
    ) -> dict[str, Any]:
        if snapshot.path is None:
            raise ValueError("Snapshot path is missing.")
        resolved_path = snapshot.path.resolve()
        try:
            relative_path = str(resolved_path.relative_to(base_dir.resolve()))
        except ValueError:
            relative_path = resolved_path.name
        return {
            "id": _snapshot_dataset_id(snapshot),
            "dataset_class_name": snapshot.dataset_class_name,
            "config_name": snapshot.config_name,
            "config_hash": snapshot.config_hash,
            "storage_type": (
                snapshot.storage_type.value if snapshot.storage_type is not None else ""
            ),
            "data_model": snapshot.data_model,
            "created_at": snapshot.created_at,
            "path": str(resolved_path),
            "relative_path": relative_path,
            "splits": snapshot.splits,
            "metadata": snapshot.metadata
            or {"description": "", "homepage": "", "license": ""},
        }

    def get_dataset(self, dataset_id: str) -> CachedDataset[Any]:
        with self.lock:
            if self.active_dataset_id == dataset_id and self.active_dataset is not None:
                return self.active_dataset

            snapshot = self._require_snapshot(dataset_id)
            if snapshot.path is None:
                raise HTTPException(
                    status_code=404,
                    detail="Prepared dataset snapshot is missing its path.",
                )

            dataset = CachedDataset(snapshot.path)
            self.active_dataset_id = dataset_id
            self.active_dataset = dataset
            return dataset

    def enqueue_job(self, request: PrepareJobRequest) -> dict[str, Any]:
        if request.name not in atria_datasets.__all__:
            raise HTTPException(status_code=404, detail="Unknown dataset.")

        source_dir = _normalize_base_dir(request.source_dir)
        if source_dir is not None and not Path(source_dir).expanduser().exists():
            raise HTTPException(
                status_code=400, detail="Source directory does not exist."
            )

        output_root = _normalize_base_dir(request.output_root)
        resolved_output_root = (
            Path(output_root).expanduser()
            if output_root is not None
            else self.resolved_base_dir()
        )
        resolved_output_root.mkdir(parents=True, exist_ok=True)
        job_id = self.store.create_job(
            request,
            normalized_source_dir=source_dir,
            resolved_output_root=str(resolved_output_root.resolve()),
        )
        self._job_queue.put(job_id)
        return self.require_job(job_id)

    def require_job(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    def _job_loop(self) -> None:
        while True:
            job_id = self._job_queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._job_queue.task_done()

    def _run_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        self.store.mark_running(job_id)
        try:
            params = dict(job["config"])
            if job["source_dir"]:
                params["data_dir"] = str(Path(job["source_dir"]).expanduser().resolve())
            dataset = getattr(atria_datasets, job["dataset_name"])(**params)
            cache_dir = Path(job["output_root"]) / job["dataset_name"]
            cached = Cacher(FileStorageType.DELTALAKE).cache(
                dataset, data_dir=str(cache_dir)
            )
        except Exception as error:  # noqa: BLE001
            self.store.mark_failed(job_id, str(error))
            return

        self.store.mark_complete(job_id, str(cached.data_dir.resolve()))
        if Path(job["output_root"]).expanduser().resolve() == self.resolved_base_dir():
            self.scan_inventory(force=True)


state = ExplorerState()
app = FastAPI(title="Dataset Manifest Explorer", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


def _split_iterator(dataset: CachedDataset[Any], split: str) -> Any:
    for split_type, iterator in dataset.split_iterators.items():
        if split_type.value == split:
            return iterator
    raise HTTPException(status_code=404, detail=f"Unknown split {split!r}.")


def _sample(dataset: CachedDataset[Any], split: str, index: int) -> Any:
    iterator = _split_iterator(dataset, split)
    if index < 0 or index >= len(iterator):
        raise HTTPException(status_code=404, detail="Document index is out of range.")
    sample = iterator[index]
    if hasattr(sample, "load") and not hasattr(sample, "visual"):
        sample = sample.load()
    return sample


def _ocr(sample: Any) -> OCRAnnotation | None:
    return sample.get_annotation_by_type(AnnotationType.ocr)


def _transcription(sample: Any, annotation: OCRAnnotation | None) -> str:
    direct = sample.get_annotation_by_type(AnnotationType.transcription)
    if direct is not None and direct.text:
        return direct.text
    if annotation is None or annotation.texts is None:
        return ""
    if annotation.levels is not None:
        pages = annotation.texts[annotation.levels == OCRLevel.page.value]
        if len(pages) and str(pages[0]).strip():
            return str(pages[0])
        lines = annotation.texts[annotation.levels == OCRLevel.line.value]
        return "\n".join(str(text) for text in lines if str(text).strip())
    return " ".join(str(text) for text in annotation.texts if str(text).strip())


def _counts(annotation: OCRAnnotation | None) -> dict[str, int]:
    if annotation is None or annotation.levels is None:
        return {}
    names = {level.value: level.name for level in OCRLevel}
    return {
        names.get(int(level), str(level)): count
        for level, count in Counter(annotation.levels.tolist()).items()
    }


def _warnings(sample: Any, annotation: OCRAnnotation | None) -> list[str]:
    result = []
    if annotation is None:
        if sample.get_annotation_by_type(AnnotationType.transcription) is None:
            result.append("Document has no OCR or transcription annotation.")
        return result
    if annotation.texts is None or not any(
        str(text).strip() for text in annotation.texts
    ):
        result.append("OCR annotation contains no transcription text.")
    if annotation.ids is not None and annotation.parent_ids is not None:
        parents = annotation.parent_ids[annotation.parent_ids != -1]
        dangling = parents[~np.isin(parents, annotation.ids)]
        if len(dangling):
            result.append(f"Found {len(dangling)} dangling parent references.")
    if annotation.bboxes is not None and (
        np.any(annotation.bboxes < 0) or np.any(annotation.bboxes > 1)
    ):
        result.append("Some bounding boxes are outside normalized image coordinates.")
    return result


def _summary(dataset_id: str, sample: Any, split: str, index: int) -> dict[str, Any]:
    annotation = _ocr(sample)
    text = _transcription(sample, annotation)
    return {
        "index": index,
        "split": split,
        "sample_id": sample.sample_id,
        "transcription_preview": text[:180],
        "annotation_counts": _counts(annotation),
        "warning_count": len(_warnings(sample, annotation)),
        "image_url": (
            f"/api/datasets/{dataset_id}/documents/{split}/{index}/image?thumbnail=true"
        ),
    }


def _document(
    dataset_id: str, dataset: CachedDataset[Any], sample: Any, split: str, index: int
) -> dict[str, Any]:
    annotation = _ocr(sample)
    loaded = sample.load()
    image = loaded.require_content()
    rows = []
    if annotation is not None and annotation.texts is not None:
        total = len(annotation.texts)
        valid_level_values = {item.value for item in OCRLevel}
        for row_index in range(total):
            level_value = (
                int(annotation.levels[row_index])
                if annotation.levels is not None
                else None
            )
            rows.append(
                {
                    "id": int(annotation.ids[row_index])
                    if annotation.ids is not None
                    else row_index,
                    "parent_id": int(annotation.parent_ids[row_index])
                    if annotation.parent_ids is not None
                    else None,
                    "level": OCRLevel(level_value).name
                    if level_value in valid_level_values
                    else level_value,
                    "text": str(annotation.texts[row_index]),
                    "bbox": annotation.bboxes[row_index].round(5).tolist()
                    if annotation.bboxes is not None
                    else None,
                }
            )
    visual_path = getattr(sample.visual, "file_path", None)
    return {
        **_summary(dataset_id, sample, split, index),
        "dataset": dataset.dataset_class_name,
        "dataset_id": dataset_id,
        "data_dir": str(dataset.data_dir.resolve()),
        "image_path": visual_path,
        "width": image.width,
        "height": image.height,
        "transcription": _transcription(sample, annotation),
        "warnings": _warnings(sample, annotation),
        "ocr_rows": rows,
        "ocr_row_count": len(rows),
        "image_url": f"/api/datasets/{dataset_id}/documents/{split}/{index}/image",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/settings")
def settings() -> dict[str, Any]:
    return {
        "base_dir": state.configured_base_dir(),
        "resolved_base_dir": str(state.resolved_base_dir().resolve()),
        "default_base_dir": str(_default_base_dir().resolve()),
        "default_cache_root": str(Path(DEFAULT_ATRIA_CACHE_DIR).expanduser()),
    }


@app.put("/api/settings")
def update_settings(request: SettingsUpdateRequest) -> dict[str, Any]:
    inventory = state.set_base_dir(request.base_dir)
    return {
        "base_dir": inventory["base_dir"],
        "resolved_base_dir": inventory["resolved_base_dir"],
        "default_base_dir": inventory["default_base_dir"],
        "default_cache_root": inventory["default_cache_root"],
    }


@app.get("/api/prepared-datasets")
def prepared_datasets(force: bool = False) -> dict[str, Any]:
    return state.scan_inventory(force=force)


@app.get("/api/preparation-options")
def preparation_options() -> dict[str, Any]:
    return {
        "datasets": [
            {
                "name": name,
                "config_fields": _factory_fields(getattr(atria_datasets, name)),
            }
            for name in sorted(atria_datasets.__all__)
        ]
    }


@app.get("/api/jobs")
def jobs() -> dict[str, Any]:
    return {"jobs": state.store.list_jobs()}


@app.post("/api/jobs")
def create_job(request: PrepareJobRequest) -> dict[str, Any]:
    return state.enqueue_job(request)


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    return state.require_job(job_id)


@app.get("/api/datasets/{dataset_id}")
def prepared_dataset(dataset_id: str) -> dict[str, Any]:
    inventory = state.scan_inventory()
    match = next(
        (item for item in inventory["datasets"] if item["id"] == dataset_id), None
    )
    if match is None:
        raise HTTPException(status_code=404, detail="Prepared dataset not found.")
    return match


@app.get("/api/datasets/{dataset_id}/documents")
def documents(
    dataset_id: str,
    split: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    search: str = "",
) -> dict[str, Any]:
    dataset = state.get_dataset(dataset_id)
    iterator = _split_iterator(dataset, split)
    search = search.strip().lower()
    if search:
        indices = []
        for index in range(len(iterator)):
            sample = _sample(dataset, split, index)
            annotation = _ocr(sample)
            if (
                search in sample.sample_id.lower()
                or search in _transcription(sample, annotation).lower()
            ):
                indices.append(index)
    else:
        indices = list(range(len(iterator)))
    start = (page - 1) * page_size
    selected = indices[start : start + page_size]
    return {
        "documents": [
            _summary(dataset_id, _sample(dataset, split, index), split, index)
            for index in selected
        ],
        "page": page,
        "page_size": page_size,
        "total": len(indices),
        "pages": max(1, (len(indices) + page_size - 1) // page_size),
    }


@app.get("/api/datasets/{dataset_id}/documents/{split}/{index}")
def document(dataset_id: str, split: str, index: int) -> dict[str, Any]:
    dataset = state.get_dataset(dataset_id)
    return _document(dataset_id, dataset, _sample(dataset, split, index), split, index)


_LEVEL_COLORS = {
    OCRLevel.page.value: "#7c3aed",
    OCRLevel.block.value: "#2563eb",
    OCRLevel.paragraph.value: "#0891b2",
    OCRLevel.line.value: "#16a34a",
    OCRLevel.word.value: "#dc2626",
}


@app.get("/api/datasets/{dataset_id}/documents/{split}/{index}/image")
def document_image(
    dataset_id: str,
    split: str,
    index: int,
    overlay: bool = False,
    levels: str = "line,word",
    thumbnail: bool = False,
) -> Response:
    dataset = state.get_dataset(dataset_id)
    sample = _sample(dataset, split, index).load()
    image = sample.require_content().convert("RGB")
    if thumbnail:
        image.thumbnail((640, 420))
    elif overlay:
        annotation = _ocr(sample)
        selected = {part.strip() for part in levels.split(",") if part.strip()}
        if annotation is not None and annotation.bboxes is not None:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for row, bbox in enumerate(annotation.bboxes):
                level = (
                    int(annotation.levels[row])
                    if annotation.levels is not None
                    else OCRLevel.word.value
                )
                if OCRLevel(level).name not in selected:
                    continue
                x1, y1, x2, y2 = bbox * np.asarray([width, height, width, height])
                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline=_LEVEL_COLORS[level],
                    width=max(2, width // 700),
                )
                if (
                    annotation.segmentations is not None
                    and annotation.segmentation_lengths is not None
                ):
                    length = int(annotation.segmentation_lengths[row])
                    if length > 1:
                        polygon = annotation.segmentations[row, :length] * np.asarray(
                            [width, height]
                        )
                        draw.line(
                            [tuple(point) for point in polygon] + [tuple(polygon[0])],
                            fill=_LEVEL_COLORS[level],
                            width=max(2, width // 700),
                        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return Response(
        output.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("explorer.backend.app:app", host="127.0.0.1", port=8000, reload=True)
