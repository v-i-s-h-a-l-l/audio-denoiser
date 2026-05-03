import gradio as gr
import requests
import tempfile, os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

os.environ["GRADIO_SHARE"] = "true"


def denoise_audio(audio_path):
    if audio_path is None:
        return None, "❌ No audio recorded."

    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{BACKEND_URL}/denoise", files={"file": ("recording.wav", f, "audio/wav")}
        )

    if response.status_code == 200:
        # Save returned audio to temp file
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.write(response.content)
        out.flush()
        return out.name, "✅ Denoising complete! Press play to hear clean audio."
    else:
        return None, f"❌ Error: {response.text}"


with gr.Blocks(title="🎙️ Real-Time Audio Denoiser") as demo:
    gr.Markdown(
        "## 🎙️ Real-Time Audio Denoiser\nRecord audio from your microphone and get a clean denoised output."
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], type="filepath", label="🎤 Record Audio"
        )

    btn = gr.Button("🚀 Denoise Audio", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    audio_output = gr.Audio(label="🔊 Clean Output Audio", type="filepath")

    btn.click(fn=denoise_audio, inputs=[audio_input], outputs=[audio_output, status])

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    show_error=True,  # 👈 shows why share link failed
)
