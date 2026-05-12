# Audio Denoiser

A comprehensive audio denoising application featuring real-time noise reduction using state-of-the-art DeepFilterNet3 model. The project includes a web interface with Gradio, a REST API backend with FastAPI, and a mobile application built with Flutter.

## 🏗️ Architecture

The project follows a microservices architecture with the following components:

- **Backend**: FastAPI server providing REST endpoints for audio denoising
- **Frontend**: Gradio web interface for real-time audio recording and denoising
- **Docker**: Containerized deployment using Docker Compose

## 🚀 Features

- **Real-time Audio Denoising**: Remove background noise from audio recordings
- **Multiple Format Support**: Accepts WAV, MP3, OGG, FLAC, WebM audio files
- **High-Quality Output**: Returns 16-bit mono WAV at 48kHz
- **Web Interface**: User-friendly Gradio interface with microphone recording
- **REST API**: Programmatic access to denoising capabilities
- **Docker Deployment**: Easy containerized setup and scaling

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)


## 🛠️ Installation & Setup

### Docker Deployment (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd audio-denoiser
```

2. Build and start the containers:
```bash
docker-compose up --build
```

3. Access the services:
- **Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

### Public Exposure with ngrok

To expose the application publicly using ngrok:

1. Install ngrok from https://ngrok.com/download and sign up for an account

2. Configure ngrok with your authtoken:
```bash
ngrok config add-authtoken <your-authtoken>
```

3. Start the Docker containers:
```bash
docker-compose up --build
```

4. In a new terminal, expose the frontend publicly:
```bash
ngrok http 7860
```

5. Share the ngrok URL (e.g., https://xxxx-xx-xx-xx-xx.ngrok-free.app) to access the application from anywhere

**Note**: When using ngrok, the frontend will be accessible publicly, but the backend will still need to be accessible. For full public access, you may need to expose both services or configure ngrok to tunnel both ports.

### Local Development

#### Backend Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install system dependencies (Linux/WSL):
```bash
sudo apt-get update
sudo apt-get install -y build-essential curl ffmpeg libsndfile1 git
```

3. Install Rust (required by deepfilternet):
```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source ~/.cargo/env
```

4. Install Python dependencies:
```bash
pip install --upgrade pip
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install deepfilternet loguru soundfile fastapi uvicorn python-multipart gradio requests
```

5. Run the backend server:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup

1. Set the backend URL environment variable:
```bash
export BACKEND_URL=http://localhost:8000  # On Windows: set BACKEND_URL=http://localhost:8000
```

2. Run the Gradio interface:
```bash
cd frontend
python app.py
```

3. Access the frontend at http://localhost:7860


## 📚 API Documentation

### Endpoints

#### POST /denoise
Upload a noisy audio file and receive a denoised version.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (audio file)

**Response:**
- Content-Type: audio/wav
- Headers: `X-Request-ID` (unique request identifier)

**Example:**
```bash
curl -X POST http://localhost:8080/denoise \
  -F "file=@noisy_audio.wav" \
  -o clean_audio.wav
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /
Root endpoint (same as health check).

### Supported Audio Formats

- WAV (audio/wav, audio/x-wav, audio/wave)
- MP3 (audio/mpeg)
- MP4 (audio/mp4)
- OGG (audio/ogg)
- FLAC (audio/flac)
- WebM (audio/webm)

### File Size Limit

Default maximum file size: 50 MB (configurable via `MAX_FILE_MB` environment variable)

## 🔧 Configuration

### Environment Variables

- `BACKEND_URL`: Backend API URL (default: http://localhost:8000)
- `MAX_FILE_MB`: Maximum upload file size in MB (default: 50)
- `UPLOAD_DIR`: Directory for uploaded files (default: /app/uploads)
- `OUTPUT_DIR`: Directory for output files (default: /app/output)

### Docker Compose Ports

- Backend: 8080 (host) → 8000 (container)
- Frontend: 7860 (host) → 7860 (container)

## 🧠 Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **DeepFilterNet3**: State-of-the-art audio denoising model
- **PyTorch**: Deep learning framework
- **Torchaudio**: Audio processing library
- **SoundFile**: Audio file I/O library
- **Uvicorn**: ASGI server

### Frontend
- **Gradio**: Python framework for building ML demos
- **Requests**: HTTP library for API calls

### Infrastructure
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration

## 📁 Project Structure

```
audio-denoiser/
├── backend/
│   ├── main.py           # FastAPI application with endpoints
│   └── denoise.py        # DeepFilterNet3 wrapper for denoising
├── frontend/
│   └── app.py            # Gradio web interface
├── mobile_app/
│   └── audio_denoiser/   # Flutter mobile application
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Multi-container orchestration
├── uploads/              # Temporary upload directory
└── output/               # Denoised audio output directory
```

## 🔍 How It Works

1. **Audio Upload**: User uploads or records audio through the Gradio interface
2. **Validation**: Backend validates file type, size, and content
3. **Processing**: Audio is loaded and resampled to 48kHz if needed
4. **Denoising**: DeepFilterNet3 model processes the audio to remove noise
5. **Output**: Clean audio is returned as a 16-bit mono WAV file

### DeepFilterNet3 Model

The backend uses DeepFilterNet3, a deep learning model specifically designed for audio denoising. Key features:

- Real-time capable
- High-quality speech enhancement
- Automatic model weight download on first run


### Scaling the Application

For production deployment:
1. Use a load balancer (nginx/traefik) for multiple backend instances
2. Implement persistent storage for uploads/output
3. Add monitoring and logging (Prometheus, Grafana)
4. Set up CI/CD pipeline


