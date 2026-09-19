#!/usr/bin/env python3
"""
StarMapr FastAPI wrapper.

Runs the existing CLI workflow behind a small async HTTP API so StarMapr can be
called remotely without replacing the current script-driven interface.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from utils import get_actor_folder_name, get_venv_python


ROOT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT_DIR / "05_videos"
JOBS_DIR = ROOT_DIR / "06_jobs"
HEADSHOT_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THUMBNAIL_PATTERNS = ("thumbnail*.jpg", "thumbnail*.jpeg", "thumbnail*.png", "*.webp")
HEADSHOT_FILENAME_RE = re.compile(r"_match_(?P<match>[0-9.]+)_position_(?P<position>\d+)")
JOB_QUEUE: Queue[str] = Queue()
JOB_LOCK = threading.Lock()
JOB_STATUS_CONDITION = threading.Condition(JOB_LOCK)
WORKER_THREAD: Optional[threading.Thread] = None


class HeadshotJobRequest(BaseModel):
    video_urls: list[str] = Field(default_factory=list)
    show: str
    actors: list[str]
    generate_thumbnail: bool = True
    verbose: bool = False


app = FastAPI(title="StarMapr API", version="1.0.0")


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def ensure_jobs_dir() -> None:
    """Create the jobs directory if it does not already exist."""
    JOBS_DIR.mkdir(exist_ok=True)


def get_job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def get_job_status_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "status.json"


def get_job_log_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "run.log"


def normalize_actors(actors: list[str]) -> list[str]:
    """Trim and deduplicate actors while preserving order."""
    normalized = []

    for actor_name in actors:
        actor_name = actor_name.strip()
        if actor_name and actor_name not in normalized:
            normalized.append(actor_name)

    return normalized


def normalize_video_urls(video_urls: list[str]) -> list[str]:
    """Trim and deduplicate candidate video URLs while preserving order."""
    normalized = []

    for video_url in video_urls:
        video_url = video_url.strip()
        if video_url and video_url not in normalized:
            normalized.append(video_url)

    return normalized


def normalize_video_folder_name(video_folder: str) -> str:
    """Normalize either 05_videos/<folder> or <folder> to the bare folder name."""
    return Path(video_folder.rstrip("/")).name


def is_terminal_status(status: str) -> bool:
    """Return whether a job status is terminal."""
    return status in {"succeeded", "failed"}


def read_job(job_id: str) -> dict[str, Any]:
    """Read a job status file from disk."""
    status_path = get_job_status_path(job_id)
    if not status_path.exists():
        raise KeyError(job_id)

    return json.loads(status_path.read_text(encoding="utf-8"))


def write_job(job_id: str, payload: dict[str, Any]) -> None:
    """Write a job status file to disk."""
    get_job_status_path(job_id).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    """Apply a partial update to a job status file."""
    with JOB_STATUS_CONDITION:
        payload = read_job(job_id)
        payload.update(changes)
        write_job(job_id, payload)
        JOB_STATUS_CONDITION.notify_all()
        return payload


def get_job_or_404(job_id: str) -> dict[str, Any]:
    """Return a persisted job payload or raise a 404."""
    try:
        with JOB_LOCK:
            return read_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


def build_command(video_url: str, request_payload: dict[str, Any]) -> list[str]:
    """Build the CLI command used to execute a headshot job."""
    command = [
        get_venv_python(),
        "01_run_headshot_detection.py",
        video_url,
        "--show", request_payload["show"],
        "--actors", ",".join(request_payload["actors"]),
        "--json",
    ]

    if request_payload.get("verbose"):
        command.append("--verbose")

    return command


def format_command(command: list[str]) -> str:
    """Format a subprocess command for log output."""
    return " ".join(shlex.quote(part) for part in command)


def parse_headshot_metadata(filename: str) -> tuple[Optional[float], Optional[int]]:
    """Extract similarity and frame position metadata from a headshot filename."""
    match = HEADSHOT_FILENAME_RE.search(filename)
    if not match:
        return None, None

    return float(match.group("match")), int(match.group("position"))


def headshot_sort_key(entry: dict[str, Any]) -> tuple[float, int, str]:
    """Sort best match first, then nearest to the middle of the sketch."""
    match_score = entry.get("match")
    position = entry.get("position")
    match_sort = -(match_score if match_score is not None else -1.0)
    position_sort = abs(position - 5000) if position is not None else 10**9
    return match_sort, position_sort, entry["filename"]


def run_command_to_log(command: list[str], log_file, job_id=None) -> tuple[int, str]:
    """Run a subprocess, stream merged output to the log, and capture the last non-empty line."""
    last_nonempty_line = ""

    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, 'PYTHONUNBUFFERED': '1', **({'STARMAPR_PROGRESS_FILE': str(get_job_log_path(job_id).with_suffix('.progress.json'))} if job_id else {})},
    )

    assert process.stdout is not None

    for line in process.stdout:
        log_file.write(line)
        log_file.flush()

        if job_id and line.startswith('STARMAPR_PROGRESS '):
            try:
                event = json.loads(line.removeprefix('STARMAPR_PROGRESS '))
                if event.get('state') != 'waiting':
                    update_job(job_id, progress=event)
            except (ValueError, TypeError):
                pass
            continue

        if line.strip():
            last_nonempty_line = line.strip()

    return process.wait(), last_nonempty_line


def build_artifact_entry(job_id: str, file_path: Path) -> dict[str, Any]:
    """Convert a repo file path into artifact metadata."""
    relative_path = file_path.relative_to(ROOT_DIR).as_posix()
    entry = {
        "filename": file_path.name,
        "relative_path": relative_path,
        "download_url": f"/jobs/{job_id}/artifacts/{relative_path}",
    }

    match_score, position = parse_headshot_metadata(file_path.name)
    if match_score is not None:
        entry["match"] = match_score
    if position is not None:
        entry["position"] = position

    return entry


def collect_headshot_artifacts(job_id: str, video_folder: str, actor_names: list[str],
                               outcomes: Optional[dict[str, Any]] = None) -> dict[str, list[dict[str, Any]]]:
    """Expose only accepted portraits listed by this job's validation results."""
    video_dir = VIDEOS_DIR / normalize_video_folder_name(video_folder)
    headshots = {}

    for actor_name in actor_names:
        actor_dir = video_dir / "headshots" / get_actor_folder_name(actor_name)
        entries = []

        report = (outcomes or {}).get(actor_name, {})
        if report.get('status') == 'accepted':
            seen_names = set()
            for headshot in report.get('headshots', []):
                filename = headshot.get('file', '')
                if not filename or Path(filename).name != filename or filename in seen_names:
                    continue
                file_path = actor_dir / filename
                if (file_path.is_file() and not file_path.is_symlink()
                        and file_path.suffix.lower() in HEADSHOT_EXTENSIONS):
                    entry = build_artifact_entry(job_id, file_path)
                    entry['validation'] = headshot.get('verification', {})
                    entries.append(entry)
                    seen_names.add(filename)

        entries.sort(key=headshot_sort_key)
        headshots[actor_name] = entries[:3]

    return headshots


def collect_thumbnail_artifacts(job_id: str, video_folder: str) -> list[dict[str, Any]]:
    """Collect generated thumbnails plus yt-dlp thumbnail fallbacks."""
    video_dir = VIDEOS_DIR / normalize_video_folder_name(video_folder)
    if not video_dir.exists():
        return []

    thumbnails = []
    seen_paths = set()

    for pattern in THUMBNAIL_PATTERNS:
        for file_path in sorted(video_dir.glob(pattern)):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(ROOT_DIR).as_posix()
            if relative_path in seen_paths:
                continue

            seen_paths.add(relative_path)
            thumbnails.append(build_artifact_entry(job_id, file_path))

    return thumbnails[:3]


def should_try_next_video_url(result_payload: dict[str, Any]) -> bool:
    """Return whether a failed job should try the next candidate video URL."""
    error_text = (result_payload.get("error") or "").lower()
    return (
        "video download failed" in error_text
        or "no headshots found for any successfully trained actors" in error_text
    )


def finalize_job_result(job_id: str, request_payload: dict[str, Any], result_payload: dict[str, Any]) -> dict[str, Any]:
    """Attach ranked candidate metadata to a raw CLI result payload."""
    video_folder = result_payload.get("video_folder")
    if not video_folder:
        result_payload["headshots"] = {actor_name: [] for actor_name in request_payload["actors"]}
        result_payload["thumbnails"] = []
    else:
        result_payload["headshots"] = collect_headshot_artifacts(
            job_id, video_folder, request_payload["actors"],
            result_payload.get('headshot_outcomes', {}) if result_payload.get('success') else {})
        result_payload["thumbnails"] = collect_thumbnail_artifacts(job_id, video_folder)

    result_payload["actors_with_headshots"] = [
        actor_name
        for actor_name, entries in result_payload["headshots"].items()
        if entries
    ]
    result_payload["actors_without_headshots"] = [
        actor_name
        for actor_name, entries in result_payload["headshots"].items()
        if not entries
    ]
    return result_payload


def run_thumbnail_generation(result_payload: dict[str, Any], log_file) -> dict[str, Any]:
    """Run the existing thumbnail script after a successful headshot job."""
    video_folder = result_payload.get("video_folder")
    if not video_folder:
        return {
            "thumbnail_command": None,
            "thumbnail_success": False,
        }

    command = [get_venv_python(), "34_extract_video_thumbnail.py", video_folder]
    log_file.write(f"\n$ {format_command(command)}\n\n")
    log_file.flush()

    exit_code, _ = run_command_to_log(command, log_file)
    return {
        "thumbnail_command": command,
        "thumbnail_success": exit_code == 0,
    }


def run_headshot_job(job_id: str) -> None:
    """Execute a queued headshot job and persist status updates."""
    request_payload = read_job(job_id)["request"]
    video_urls = request_payload["video_urls"]

    update_job(
        job_id,
        status="running",
        started_at=utc_now(),
        command=build_command(video_urls[0], request_payload),
        error=None,
    )

    exit_code = 1
    result_payload = None
    attempts = []

    try:
        with get_job_log_path(job_id).open("a", encoding="utf-8") as log_file:
            for attempt_number, video_url in enumerate(video_urls, start=1):
                command = build_command(video_url, request_payload)
                update_job(job_id, command=command)

                log_file.write(f"\n=== VIDEO ATTEMPT {attempt_number}/{len(video_urls)} ===\n")
                log_file.write(f"video_url: {video_url}\n")
                log_file.write(f"$ {format_command(command)}\n\n")
                log_file.flush()

                exit_code, last_nonempty_line = run_command_to_log(command, log_file, job_id)

                if last_nonempty_line:
                    try:
                        result_payload = json.loads(last_nonempty_line)
                    except json.JSONDecodeError:
                        result_payload = None
                else:
                    result_payload = None

                if result_payload is None:
                    result_payload = {
                        "success": False,
                        "error": "Headshot command exited without JSON output",
                    }

                result_payload["attempted_video_url"] = video_url
                attempts.append({
                    "attempt_number": attempt_number,
                    "video_url": video_url,
                    "exit_code": exit_code,
                    "success": result_payload.get("success", exit_code == 0),
                    "error": result_payload.get("error"),
                    "video_folder": result_payload.get("video_folder"),
                })

                if request_payload.get("generate_thumbnail") and result_payload.get("success"):
                    result_payload.update(run_thumbnail_generation(result_payload, log_file))

                result_payload["attempts"] = attempts
                finalize_job_result(job_id, request_payload, result_payload)

                if result_payload.get("success"):
                    break

                if attempt_number < len(video_urls) and should_try_next_video_url(result_payload):
                    log_file.write("\n[retrying next candidate video URL]\n")
                    log_file.flush()
                    continue

                break

        update_job(
            job_id,
            status="succeeded" if result_payload and result_payload.get("success") else "failed",
            finished_at=utc_now(),
            exit_code=exit_code,
            result=result_payload,
            error=(
                None
                if result_payload and result_payload.get("success")
                else (result_payload or {}).get("error", f"Command exited with code {exit_code}")
            ),
        )
    except Exception as exc:
        with get_job_log_path(job_id).open("a", encoding="utf-8") as log_file:
            log_file.write("\n[server exception]\n")
            log_file.write(traceback.format_exc())

        update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            exit_code=exit_code,
            result=result_payload,
            error=str(exc),
        )


def worker_loop() -> None:
    """Process headshot jobs serially."""
    while True:
        job_id = JOB_QUEUE.get()
        try:
            run_headshot_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def ensure_worker_started() -> None:
    """Start the single background worker exactly once."""
    global WORKER_THREAD

    if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
        return

    WORKER_THREAD = threading.Thread(target=worker_loop, name="starmapr-api-worker", daemon=True)
    WORKER_THREAD.start()


def iter_artifact_entries(artifacts: dict[str, Any]):
    """Yield all artifact entries from a job result."""
    for actor_entries in artifacts.get("headshots", {}).values():
        for entry in actor_entries:
            yield entry

    for entry in artifacts.get("thumbnails", []):
        yield entry


@app.on_event("startup")
def startup_event() -> None:
    ensure_jobs_dir()
    ensure_worker_started()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/jobs/headshots")
def create_headshot_job(request: HeadshotJobRequest) -> dict[str, str]:
    ensure_jobs_dir()
    ensure_worker_started()

    video_urls = normalize_video_urls(request.video_urls)
    actors = normalize_actors(request.actors)
    show_name = request.show.strip()

    if not video_urls:
        raise HTTPException(status_code=422, detail="video_urls must contain at least one non-empty URL")

    if not show_name:
        raise HTTPException(status_code=422, detail="show must not be blank")

    if not actors:
        raise HTTPException(status_code=422, detail="actors must contain at least one non-empty actor name")

    job_id = uuid.uuid4().hex
    get_job_dir(job_id).mkdir(parents=True, exist_ok=False)
    get_job_log_path(job_id).write_text("", encoding="utf-8")

    write_job(job_id, {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "request": {
            "video_urls": video_urls,
            "show": show_name,
            "actors": actors,
            "generate_thumbnail": request.generate_thumbnail,
            "verbose": request.verbose,
        },
        "command": None,
        "exit_code": None,
        "result": None,
        "error": None,
    })

    with JOB_STATUS_CONDITION:
        JOB_STATUS_CONDITION.notify_all()
    JOB_QUEUE.put(job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/jobs/{job_id}",
        "wait_url": f"/jobs/{job_id}/wait",
        "log_url": f"/jobs/{job_id}/log",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return get_job_or_404(job_id)


@app.get("/jobs/{job_id}/wait")
def wait_for_job(job_id: str, timeout: int = 900) -> dict[str, Any]:
    deadline = time.monotonic() + max(timeout, 0)

    with JOB_STATUS_CONDITION:
        while True:
            try:
                payload = read_job(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

            if is_terminal_status(payload.get("status", "")):
                return payload

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return payload

            JOB_STATUS_CONDITION.wait(timeout=min(remaining, 5))


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_job_log(job_id: str, tail: Optional[int] = None) -> str:
    get_job_or_404(job_id)
    log_path = get_job_log_path(job_id)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log not found for job: {job_id}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if tail and tail > 0:
        log_text = "\n".join(log_text.splitlines()[-tail:])

    return log_text


@app.get("/jobs/{job_id}/artifacts/{artifact_path:path}")
def download_artifact(job_id: str, artifact_path: str):
    job_payload = get_job_or_404(job_id)
    result_payload = job_payload.get("result") or {}
    candidates = {
        "headshots": result_payload.get("headshots", {}),
        "thumbnails": result_payload.get("thumbnails", []),
    }
    allowed_paths = {entry["relative_path"] for entry in iter_artifact_entries(candidates)}

    if artifact_path not in allowed_paths:
        raise HTTPException(status_code=404, detail=f"Artifact not found for job: {artifact_path}")

    file_path = ROOT_DIR / artifact_path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact file not found: {artifact_path}")

    return FileResponse(file_path, filename=file_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StarMapr FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
