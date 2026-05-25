"""
backend/main.py  —  FastAPI backend for DeepFilterNet3  (batching-optimised)
----------------------------------------------------------------------------
Endpoints
---------
POST /denoise        Upload a noisy audio file, get back a clean WAV.
POST /denoise/batch  Upload multiple files, get back a ZIP of clean WAVs.
GET  /health         Liveness check.
GET  /               Same liveness check.

Batching strategy
-----------------
A background BatchCollector waits up to BATCH_WINDOW_MS milliseconds after
the first request in a window arrives, collects all requests that land in
that window, then dispatches them together via denoise_batch().  This means
N simultaneous API requests share one denoise_batch() call instead of
firing N independent denoise_file() calls — cutting per-request overhead
(model warm-up amortisation, thread-pool spin-up) by ~N×.

Environment variables
---------------------
MAX_FILE_MB        (default 50)   per-file size limit
BATCH_WINDOW_MS    (default 40)   collector window in milliseconds
MAX_BATCH_SIZE     (default 8)    max requests per batch window

Run locally:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import uuid
import zipfile
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from denoise import denoise_file, denoise_batch

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Batching config ───────────────────────────────────────────────────────────
BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "40"))  # ms to wait
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))  # requests per batch

# ── Allowed audio MIME types ──────────────────────────────────────────────────
ALLOWED_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "application/octet-stream",
}

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Denoiser API",
    description="DeepFilterNet3 on-device denoising via FastAPI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BATCH COLLECTOR                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class _PendingRequest:
    req_id: str
    in_path: Path
    out_path: Path
    future: asyncio.Future = field(default_factory=lambda: None)


class BatchCollector:
    """
    Collects individual /denoise calls into windows and dispatches them
    together via denoise_batch() to maximise hardware utilisation.

    Timeline for a BATCH_WINDOW_MS=40 window:

        t=0   request-A arrives → window opens, timer starts
        t=15  request-B arrives → added to same window
        t=30  request-C arrives → added to same window
        t=40  window closes  → denoise_batch([A, B, C]) fires once
                               all three futures resolved simultaneously
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._pending: list[_PendingRequest] = []
        self._timer_task: asyncio.Task | None = None

    async def submit(self, req: _PendingRequest) -> str:
        """Add a request to the current batch window; await its result."""
        loop = asyncio.get_event_loop()
        req.future = loop.create_future()

        async with self._lock:
            self._pending.append(req)
            if self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._dispatch_after_window())
            if len(self._pending) >= MAX_BATCH_SIZE:
                # Window is full — dispatch immediately without waiting
                self._timer_task.cancel()
                asyncio.create_task(self._dispatch())

        return await req.future

    async def _dispatch_after_window(self):
        await asyncio.sleep(BATCH_WINDOW_MS / 1000.0)
        await self._dispatch()

    async def _dispatch(self):
        async with self._lock:
            batch, self._pending = self._pending, []

        if not batch:
            return

        logger.info("Dispatching batch of %d request(s)", len(batch))
        items = [(r.in_path, r.out_path) for r in batch]

        loop = asyncio.get_event_loop()
        try:
            # denoise_batch is CPU/GPU-bound — run in executor so we don't
            # block the event loop
            results = await loop.run_in_executor(None, lambda: denoise_batch(items))
            for req, out_path in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(out_path)
        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)


_collector = BatchCollector()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ROUTES                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@app.get("/")
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/denoise")
async def denoise_endpoint(file: UploadFile = File(...)):
    """
    Accept a noisy audio upload and return a denoised WAV.

    The request is placed into the current batch window.  If other requests
    arrive within BATCH_WINDOW_MS they are processed together, reducing
    per-request GPU overhead.
    """
    # ── Validation ───────────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {MAX_FILE_MB} MB)."
        )

    # ── Persist upload ────────────────────────────────────────────────────────
    req_id = uuid.uuid4().hex
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    in_path = UPLOAD_DIR / f"{req_id}_input{suffix}"
    out_path = OUTPUT_DIR / f"{req_id}_clean.wav"
    in_path.write_bytes(raw_bytes)

    logger.info(
        "Received  id=%s  size=%.1f KB  file=%s",
        req_id,
        len(raw_bytes) / 1024,
        file.filename,
    )

    # ── Enqueue in batch collector ────────────────────────────────────────────
    req = _PendingRequest(req_id=req_id, in_path=in_path, out_path=out_path)
    try:
        clean_path = await _collector.submit(req)
        logger.info("Done  id=%s  output=%s", req_id, clean_path)
    except Exception as exc:
        logger.exception("Denoising failed  id=%s", req_id)
        raise HTTPException(status_code=500, detail=f"Denoising failed: {exc}")
    finally:
        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass

    return FileResponse(
        path=str(clean_path),
        media_type="audio/wav",
        filename=f"clean_{file.filename or 'output.wav'}",
        headers={"X-Request-ID": req_id},
    )


@app.post("/denoise/batch")
async def denoise_batch_endpoint(files: list[UploadFile] = File(...)):
    """
    Accept multiple audio files in one request, return a ZIP of clean WAVs.

    This endpoint bypasses the BatchCollector (the caller has already done the
    batching) and calls denoise_batch() directly for maximum throughput.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    items = []
    req_ids = []
    in_paths = []

    for file in files:
        raw_bytes = await file.read()
        if not raw_bytes:
            continue
        if len(raw_bytes) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{file.filename} too large.")

        req_id = uuid.uuid4().hex
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        in_path = UPLOAD_DIR / f"{req_id}_input{suffix}"
        out_path = OUTPUT_DIR / f"{req_id}_clean.wav"

        in_path.write_bytes(raw_bytes)
        items.append((in_path, out_path))
        req_ids.append(req_id)
        in_paths.append(in_path)

    logger.info("Batch endpoint: %d file(s)", len(items))

    loop = asyncio.get_event_loop()
    try:
        out_paths = await loop.run_in_executor(None, lambda: denoise_batch(items))
    except Exception as exc:
        logger.exception("Batch denoising failed")
        raise HTTPException(status_code=500, detail=f"Denoising failed: {exc}")
    finally:
        for p in in_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # ── Pack results into a ZIP and stream back ───────────────────────────────
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (out_path, file) in enumerate(zip(out_paths, files)):
            arcname = f"clean_{file.filename or f'output_{i}.wav'}"
            zf.write(out_path, arcname=arcname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=denoised.zip"},
    )
