from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil, uuid
from denoise import denoise_file

import sys, os

sys.path.append(os.path.dirname(__file__))

app = FastAPI()

UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...)):
    # Save uploaded file
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    output_path = OUTPUT_DIR / f"{job_id}_clean.wav"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run denoising
    denoise_file(input_path, output_path)

    return FileResponse(
        path=str(output_path), media_type="audio/wav", filename="clean_output.wav"
    )


@app.get("/health")
def health():
    return {"status": "ok"}
