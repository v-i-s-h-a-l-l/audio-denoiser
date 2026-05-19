"""
denoise.py  —  DeepFilterNet3 audio denoising utility
------------------------------------------------------
Used by backend/main.py:

    from denoise import denoise_file

    clean_path = denoise_file(input_path, output_path)

Requirements (already in your Dockerfile):
    deepfilternet, torch, torchaudio, soundfile, soxr
"""

import os
import logging
from pathlib import Path

import torch
import numpy as np
import soxr
import soundfile as sf
from df.enhance import enhance, init_df, load_audio, save_audio

logger = logging.getLogger(__name__)

TARGET_SR = 16_000  # output sample rate after resampling

# ── Model is loaded once at import time so every request reuses it ──────────
_model = None
_df_state = None
_model_sr = None  # native sample-rate the model expects (usually 48 000 Hz)


def _load_model():
    global _model, _df_state, _model_sr
    if _model is None:
        logger.info("Loading DeepFilterNet3 model (first call)…")
        _model, _df_state, _ = init_df()
        _model_sr = _df_state.sr()
        _model.eval()
        logger.info("DeepFilterNet3 ready  (sr=%d Hz)", _model_sr)
    return _model, _df_state, _model_sr


def _resample_to_16k(audio: torch.Tensor, from_sr: int) -> np.ndarray:
    """
    Resample a (1, samples) torch float32 tensor from `from_sr` to TARGET_SR
    using soxr (high-quality resampler).

    Returns a (samples,) float32 numpy array ready for soundfile.
    """
    if from_sr == TARGET_SR:
        return audio.squeeze(0).numpy()

    mono_np = audio.squeeze(0).numpy()  # (samples,)  float32
    resampled = soxr.resample(
        mono_np,
        in_rate=from_sr,
        out_rate=TARGET_SR,
        quality="HQ",  # HQ = high quality, good balance
    )
    return resampled.astype(np.float32)


# ── Public API ───────────────────────────────────────────────────────────────


def denoise_file(input_path: str | os.PathLike, output_path: str | os.PathLike) -> str:
    """
    Denoise a WAV (or any audio file readable by torchaudio), resample to
    16 kHz via soxr, and write the result to *output_path* as a 16-bit
    mono WAV.

    Parameters
    ----------
    input_path  : path to the noisy recording
    output_path : where to write the cleaned WAV

    Returns
    -------
    str  –  absolute path to the cleaned WAV (same as output_path)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    model, df_state, model_sr = _load_model()

    # ── 1. Load audio ────────────────────────────────────────────────────────
    logger.info("Loading audio: %s", input_path)
    audio, meta = load_audio(str(input_path), sr=model_sr)

    logger.info(
        "  duration=%.2f s  channels=%d  sr=%d",
        audio.shape[-1] / model_sr,
        audio.shape[0],
        model_sr,
    )

    # ── 2. Collapse to mono if needed ────────────────────────────────────────
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # ── 3. Run DeepFilterNet3 ────────────────────────────────────────────────
    logger.info("Running DeepFilterNet3…")
    with torch.no_grad():
        enhanced = enhance(model, df_state, audio)
    # enhanced: (1, samples) float32 at model_sr (48 kHz)

    # ── 4. Resample to 16 kHz via soxr ──────────────────────────────────────
    logger.info("Resampling %d Hz → %d Hz via soxr…", model_sr, TARGET_SR)
    resampled = _resample_to_16k(enhanced, from_sr=model_sr)
    # resampled: (samples,) float32 at 16 kHz

    # ── 5. Write output WAV ──────────────────────────────────────────────────
    logger.info("Writing output: %s  (sr=%d Hz)", output_path, TARGET_SR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), resampled, TARGET_SR, subtype="PCM_16")

    return str(output_path)


def denoise_bytes(audio_bytes: bytes, suffix: str = ".wav") -> bytes:
    """
    Convenience wrapper: accepts raw audio bytes, returns denoised WAV bytes
    resampled to 16 kHz.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(suffix, "_clean.wav")

    try:
        denoise_file(tmp_in_path, tmp_out_path)
        with open(tmp_out_path, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.remove(p)
            except OSError:
                pass
