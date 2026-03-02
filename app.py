"""
Whisper API Server - GPU-accelerated speech-to-text
"""
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Initialize model (lazy load)
whisper_model = None

def get_model():
    """Lazy load the Whisper model"""
    global whisper_model
    if whisper_model is None:
        logger.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        logger.info("Model loaded successfully")
    return whisper_model

app = FastAPI(
    title="Whisper API",
    description="GPU-accelerated speech-to-text using Faster-Whisper",
    version="1.0.0"
)


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=MODEL_SIZE,
        device=DEVICE
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'es'). Auto-detected if not provided."),
    task: str = Query("transcribe", description="Task: 'transcribe' or 'translate'")
):
    """
    Transcribe audio file to text
    
    Supports: WAV, MP3, OGG, M4A, FLAC, etc.
    """
    # Validate file type
    allowed_types = [
        "audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/x-m4a", "audio/mp4", "audio/flac", "audio/x-wav",
        "video/ogg", "application/ogg"
    ]
    
    # Check by extension if content-type is not set
    content_type = file.content_type or "application/octet-stream"
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    allowed_extensions = [".wav", ".mp3", ".ogg", ".m4a", ".flac", ".oga", ".opus"]
    
    if content_type not in allowed_types and ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type} ({ext})"
        )
    
    # Save uploaded file to temp
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Transcribing file: {file.filename} ({len(content)} bytes)")
        
        # Get model and transcribe
        model = get_model()
        
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            task=task
        )
        
        # Collect all segments
        text = " ".join(segment.text.strip() for segment in segments)
        
        logger.info(f"Transcription complete: {len(text)} chars, language: {info.language}")
        
        return TranscriptionResponse(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            model=MODEL_SIZE
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/transcribe/url")
async def transcribe_from_url(
    url: str = Query(..., description="URL to audio file"),
    language: Optional[str] = Query(None, description="Language code")
):
    """
    Transcribe audio from URL
    """
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        
        # Determine extension from URL or content-type
        ext = Path(url).suffix.lower() or ".mp3"
        content_type = response.headers.get("content-type", "")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Transcribe
        model = get_model()
        segments, info = model.transcribe(
            tmp_path,
            language=language
        )
        
        text = " ".join(segment.text.strip() for segment in segments)
        
        return TranscriptionResponse(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            model=MODEL_SIZE
        )
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
