from pathlib import Path
import torchaudio
from df.enhance import enhance, init_df

# Load model once at startup
model, df_state, _ = init_df()


def denoise_file(input_path: Path, output_path: Path):
    audio, sr = torchaudio.load(str(input_path))

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if sr != 48000:
        audio = torchaudio.functional.resample(audio, sr, 48000)
        sr = 48000

    enhanced = enhance(model, df_state, audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), enhanced, sr)
    return output_path
