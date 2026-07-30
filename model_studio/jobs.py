"""Background run management and durable result artifacts."""

from __future__ import annotations

import csv
import json
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .domain import RunResult


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "Queued"
    completed: int = 0
    total: int = 0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result_path: str | None = None
    csv_path: str | None = None
    _cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    def public(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_path": self.result_path,
            "csv_path": self.csv_path,
        }
        payload["progress"] = (
            self.completed / self.total if self.total else 0.0
        )
        return payload


class JobManager:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )
            return [job.public() for job in jobs]

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        job._cancel_event.set()
        return job.public()

    def submit(self, task: Callable[[Callable, threading.Event], RunResult]) -> Job:
        job = Job(id=f"run-{uuid.uuid4().hex[:10]}")
        with self._lock:
            self._jobs[job.id] = job

        def progress(completed: int, total: int, stage: str) -> None:
            with self._lock:
                job.completed = completed
                job.total = total
                job.stage = stage

        def worker() -> None:
            job.status = "running"
            job.stage = "Baseline inference"
            job.started_at = _now()
            try:
                result = task(progress, job._cancel_event)
                run_dir = self.runs_root / job.id
                run_dir.mkdir(parents=True, exist_ok=True)
                result_path = run_dir / "result.json"
                result_path.write_text(json.dumps(result.to_dict()))
                csv_path = run_dir / "deletion_effects.csv"
                with csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "position",
                            "deletion_effect",
                            "plot_effect_pct",
                        ]
                    )
                    writer.writerows(
                        zip(
                            result.nucleotide_positions,
                            result.nucleotide_effects,
                            result.plot_effects,
                            strict=True,
                        )
                    )
                job.result_path = str(result_path)
                job.csv_path = str(csv_path)
                job.completed = max(job.completed, job.total)
                job.status = "complete"
                job.stage = "Complete"
            except InterruptedError:
                job.status = "cancelled"
                job.stage = "Cancelled"
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.stage = "Failed"
                job.error = f"{exc}\n{traceback.format_exc(limit=8)}"
            finally:
                job.finished_at = _now()

        threading.Thread(target=worker, daemon=True).start()
        return job
