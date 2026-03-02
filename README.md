# Whisper API

GPU-accelerated speech-to-text API using Faster-Whisper.

## Features

- 🚀 GPU-accelerated with CUDA
- 🎯 Multiple model sizes (tiny, base, small, medium, large-v3)
- 🌍 Auto language detection
- 🔄 Supports multiple audio formats (WAV, MP3, OGG, M4A, FLAC)
- 🐳 Docker container with GPU passthrough
- ⚡ FastAPI for fast HTTP requests

## Quick Start

### Docker Compose

```yaml
version: '3.8'

services:
  whisper-api:
    image: ghcr.io/ellickjohnson/whisper-api:latest
    container_name: whisper-api
    restart: unless-stopped
    ports:
      - "8001:8001"
    environment:
      - WHISPER_MODEL=medium
      - WHISPER_DEVICE=cuda
      - WHISPER_COMPUTE_TYPE=float16
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Usage

**Health Check:**
```bash
curl http://localhost:8001/health
```

**Transcribe Audio:**
```bash
curl -X POST "http://localhost:8001/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.ogg"
```

**Response:**
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 3.5,
  "model": "medium"
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WHISPER_MODEL` | `medium` | Model size (tiny, base, small, medium, large-v3) |
| `WHISPER_DEVICE` | `cuda` | Device (cuda, cpu) |
| `WHISPER_COMPUTE_TYPE` | `float16` | Compute type (float16, float32, int8) |

## Model Sizes & VRAM

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| `tiny` | ~1GB | Very Fast | Decent |
| `base` | ~1GB | Fast | Good |
| `small` | ~2GB | Fast | Better |
| `medium` | ~5GB | Medium | Great |
| `large-v3` | ~10GB | Slower | Best |

## GPU Sharing

Whisper can share GPU with other containers (like Ollama) as long as:
- Total VRAM usage doesn't exceed GPU capacity
- Models are loaded on-demand

Example with 12GB GPU:
- Whisper `medium` (5GB) + Ollama 7B model (5GB) = 10GB ✅
- Whisper `large-v3` (10GB) + Ollama 14B model (9GB) = 19GB ❌

## API Endpoints

### `GET /health`
Health check endpoint.

### `POST /transcribe`
Transcribe audio file.

**Parameters:**
- `file` (required): Audio file
- `language` (optional): Language code (e.g., 'en', 'es')
- `task` (optional): 'transcribe' or 'translate'

### `POST /transcribe/url`
Transcribe audio from URL.

**Parameters:**
- `url` (required): URL to audio file
- `language` (optional): Language code

## Integration with OpenClaw

This API is designed to work with OpenClaw for voice message transcription:

1. User sends voice message to Telegram
2. OpenClay receives audio file
3. OpenClaw calls Whisper API
4. Whisper returns transcription
5. OpenClaw processes the text

## Building

```bash
docker build -t whisper-api .
```

## License

MIT
