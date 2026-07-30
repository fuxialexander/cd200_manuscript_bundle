"""Standard-library HTTP server for the local model studio."""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import shutil
import sys
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .service import ModelStudioService


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"


class ModelStudioHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, ModelStudioHandler)
        self.service = ModelStudioService(APP_ROOT)


class ModelStudioHandler(SimpleHTTPRequestHandler):
    server: ModelStudioHTTPServer
    server_version = "CD200ModelStudio/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[model-studio] {format % args}\n")

    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlparse(path)
        relative = parsed.path.lstrip("/") or "index.html"
        target = (STATIC_ROOT / relative).resolve()
        if not target.is_relative_to(STATIC_ROOT.resolve()):
            return str(STATIC_ROOT / "404")
        return str(target)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error_json(self, status: int, exc: Exception) -> None:
        self._json(
            {
                "error": type(exc).__name__,
                "message": str(exc),
            },
            status=status,
        )

    def _text(
        self,
        body: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        if download_name:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{download_name}"',
            )
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _receive_upload(
        self,
        field_name: str,
        *,
        allowed_suffixes: set[str],
        prefix: str,
    ) -> Path:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        if field_name not in form:
            raise ValueError(f"No {field_name} file was uploaded")
        item = form[field_name]
        if not getattr(item, "filename", None):
            raise ValueError(f"No {field_name} file was uploaded")
        source_name = Path(Path(item.filename).name)
        suffix = source_name.suffix.lower()
        if suffix not in allowed_suffixes:
            allowed = ", ".join(sorted(allowed_suffixes))
            raise ValueError(f"Uploaded file must use one of: {allowed}")
        upload_dir = APP_ROOT / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / (
            f"{prefix}-{source_name.stem}-{uuid.uuid4().hex[:8]}{suffix}"
        )
        with target.open("wb") as handle:
            shutil.copyfileobj(item.file, handle)
        return target.resolve()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/capabilities":
                self._json(self.server.service.capabilities())
                return
            if parsed.path == "/api/genome/mm10.chrom.sizes":
                query = urllib.parse.parse_qs(parsed.query)
                preset_id = query.get(
                    "preset_id",
                    ["manuscript_differential_promoter"],
                )[0]
                sizes = self.server.service.chromosome_sizes(preset_id)
                self._text(
                    "".join(
                        f"{chromosome}\t{length}\n"
                        for chromosome, length in sizes
                    )
                )
                return
            if parsed.path == "/api/tracks/observed-atac.bedgraph":
                query = urllib.parse.parse_qs(parsed.query)
                preset_id = query.get(
                    "preset_id",
                    ["manuscript_differential_promoter"],
                )[0]
                chromosome = query.get("chr", [""])[0]
                start = int(query.get("start", ["0"])[0])
                end = int(query.get("end", ["0"])[0])
                max_points = int(query.get("max_points", ["50000"])[0])
                body, metadata = self.server.service.observed_atac_bedgraph(
                    preset_id,
                    chromosome,
                    start,
                    end,
                    max_points=max_points,
                )
                self._text(
                    body,
                    headers={
                        "X-Track-Bin-Bp": str(metadata["bin_bp"]),
                        "X-Track-Start": str(metadata["start"]),
                        "X-Track-End": str(metadata["end"]),
                    },
                )
                return
            if parsed.path == "/api/model":
                self._json(
                    {
                        "model": self.server.service.model_metadata(),
                        "profile_model": (
                            self.server.service.profile_model_metadata()
                        ),
                    }
                )
                return
            if parsed.path == "/api/jobs":
                self._json({"jobs": self.server.service.jobs.list()})
                return
            if parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.strip("/").split("/")
                job_id = parts[2] if len(parts) > 2 else ""
                job = self.server.service.jobs.get(job_id)
                if len(parts) == 4 and parts[3] == "result":
                    self._json(self.server.service.result(job_id))
                    return
                if len(parts) == 4 and parts[3] == "result.json":
                    if not job.result_path:
                        raise RuntimeError("Result is not available")
                    self._send_file(Path(job.result_path), f"{job_id}-result.json")
                    return
                if len(parts) == 4 and parts[3] == "effects.csv":
                    if not job.csv_path:
                        raise RuntimeError("CSV is not available")
                    self._send_file(Path(job.csv_path), f"{job_id}-effects.csv")
                    return
                self._json(job.public())
                return
        except KeyError as exc:
            self._error_json(404, exc)
            return
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            self._error_json(400, exc)
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/model/load":
                payload = self._read_json()
                metadata = self.server.service.load_model(
                    architecture_id=payload.get("architecture_id", ""),
                    checkpoint=payload.get("checkpoint", ""),
                    requested_device=payload.get("device", "auto"),
                    options=payload.get("options") or {},
                    preset_id=payload.get("preset_id", "custom"),
                    profile_architecture_id=payload.get(
                        "profile_architecture_id"
                    ),
                    profile_checkpoint=payload.get("profile_checkpoint"),
                    profile_options=payload.get("profile_options") or {},
                )
                self._json(
                    {
                        "model": metadata.to_dict(),
                        "profile_model": (
                            self.server.service.profile_model_metadata()
                        ),
                    }
                )
                return
            if parsed.path == "/api/uploads/checkpoint":
                target = self._receive_upload(
                    "checkpoint",
                    allowed_suffixes={".ckpt", ".pt", ".pth"},
                    prefix="checkpoint",
                )
                self._json({"path": str(target.resolve())}, status=201)
                return
            if parsed.path == "/api/uploads/table":
                target = self._receive_upload(
                    "table",
                    allowed_suffixes={".bed", ".csv", ".tsv", ".txt"},
                    prefix="table",
                )
                self._json({"path": str(target)}, status=201)
                return
            if parsed.path == "/api/sequence/inspect":
                self._json(
                    {"sequence": self.server.service.inspect_sequence(self._read_json())}
                )
                return
            if parsed.path == "/api/annotations/load":
                self._json(
                    {
                        "annotations": self.server.service.load_annotations(
                            self._read_json()
                        )
                    }
                )
                return
            if parsed.path == "/api/annotations/filter":
                self._json(
                    {
                        "annotations": self.server.service.filter_annotations(
                            self._read_json()
                        )
                    }
                )
                return
            if parsed.path == "/api/runs":
                self._json(
                    {"job": self.server.service.submit_run(self._read_json())},
                    status=201,
                )
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                parts = parsed.path.strip("/").split("/")
                self._json({"job": self.server.service.jobs.cancel(parts[2])})
                return
            self.send_error(404)
        except KeyError as exc:
            self._error_json(404, exc)
        except (
            AttributeError,
            ImportError,
            json.JSONDecodeError,
            ValueError,
            RuntimeError,
            FileNotFoundError,
        ) as exc:
            self._error_json(400, exc)
        except Exception as exc:  # noqa: BLE001
            self._error_json(500, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Keep 127.0.0.1 unless remote access is intentional.",
    )
    parser.add_argument("--port", type=int, default=8787)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ModelStudioHTTPServer((args.host, args.port))
    print(f"CD200 Model Studio: http://{args.host}:{args.port}")
    print(f"Application root: {APP_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
