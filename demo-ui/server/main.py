"""
FastAPI server for TinyML sEMG Gesture Recognition Demo

This server provides:
1. WebSocket endpoint for real-time gesture predictions
2. REST endpoint for file upload and batch processing
3. Demo mode with simulated EMG data
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Optional
from enum import IntEnum

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="Perpetuality EMG Demo API",
    description="Real-time sEMG gesture recognition demo server",
    version="1.0.0"
)

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GestureType(IntEnum):
    """Gesture types matching the model output"""
    INDEX_PRESS = 0
    INDEX_RELEASE = 1
    MIDDLE_PRESS = 2
    MIDDLE_RELEASE = 3
    THUMB_CLICK = 4
    THUMB_DOWN = 5
    THUMB_IN = 6
    THUMB_OUT = 7
    THUMB_UP = 8


class PredictionResult(BaseModel):
    """Model prediction result"""
    timestamp: float
    emg_data: list[list[float]]  # 7 channels
    predictions: list[float]  # 9 gesture probabilities
    detected_gesture: Optional[int] = None
    confidence: float = 0.0
    latency_ms: float = 0.0


class DemoConfig(BaseModel):
    """Demo configuration"""
    sample_rate: int = 2000
    window_size: int = 10000
    output_rate: int = 200
    num_channels: int = 7
    num_gestures: int = 9


# Global state
demo_config = DemoConfig()
model = None
is_model_loaded = False


def load_model():
    """Load the trained model"""
    global model, is_model_loaded

    try:
        import torch
        from generic_neuromotor_interface.networks_isolated import FixedDiscreteGesturesLSTM

        model_path = Path(__file__).parent.parent.parent / "models_gpu_multi_20251124_121916" / "m1_best.pt"

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')

            model = FixedDiscreteGesturesLSTM(
                in_channels=7,
                out_channels=9,
                hidden_size=128,
                num_lstm_layers=3,
                dropout=0.1
            )

            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            is_model_loaded = True
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Model not found at {model_path}, running in demo mode")
            is_model_loaded = False

    except Exception as e:
        print(f"Failed to load model: {e}")
        is_model_loaded = False


def generate_demo_emg(num_samples: int = 1000) -> np.ndarray:
    """Generate simulated EMG data for demo purposes"""
    t = np.linspace(0, num_samples / 2000, num_samples)

    # Base signal: combination of noise and muscle activation patterns
    channels = []
    for i in range(7):
        # Different frequency components per channel
        base_freq = 20 + i * 5
        signal = (
            np.random.randn(num_samples) * 20 +  # Noise
            np.sin(2 * np.pi * base_freq * t) * 30 +  # Base frequency
            np.sin(2 * np.pi * (base_freq * 2) * t) * 15  # Harmonic
        )

        # Add occasional bursts (simulating muscle activation)
        if np.random.random() > 0.7:
            burst_start = np.random.randint(0, num_samples - 200)
            burst = np.exp(-((np.arange(200) - 100) ** 2) / 500) * 100
            signal[burst_start:burst_start + 200] += burst

        channels.append(signal)

    return np.array(channels)


def generate_demo_predictions() -> tuple[list[float], int | None, float]:
    """Generate simulated gesture predictions for demo"""
    # Most of the time, no gesture
    if np.random.random() > 0.15:
        return [0.05] * 9, None, 0.0

    # Random gesture with high confidence
    gesture_idx = np.random.randint(0, 9)
    predictions = [0.05] * 9
    predictions[gesture_idx] = 0.85 + np.random.random() * 0.14

    return predictions, gesture_idx, predictions[gesture_idx] * 100


@app.on_event("startup")
async def startup():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Perpetuality EMG Demo API",
        "version": "1.0.0",
        "model_loaded": is_model_loaded,
        "config": demo_config.dict()
    }


@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        "status": "ok",
        "model_loaded": is_model_loaded,
        "timestamp": time.time()
    }


@app.get("/api/config")
async def get_config():
    """Get demo configuration"""
    return demo_config.dict()


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gesture streaming

    Sends predictions at ~20Hz (50ms intervals) for smooth visualization
    """
    await websocket.accept()

    try:
        sample_counter = 0

        while True:
            start_time = time.time()

            # Generate or process EMG data
            emg_data = generate_demo_emg(100)  # 50ms of data at 2kHz

            # Get predictions
            if is_model_loaded and model is not None:
                import torch
                with torch.no_grad():
                    input_tensor = torch.tensor(emg_data, dtype=torch.float32).unsqueeze(0)
                    output = model(input_tensor)
                    predictions = torch.sigmoid(output).squeeze().tolist()

                    # Find max prediction
                    max_idx = np.argmax(predictions)
                    confidence = predictions[max_idx] * 100
                    detected = max_idx if confidence > 50 else None
            else:
                predictions, detected, confidence = generate_demo_predictions()

            latency = (time.time() - start_time) * 1000

            # Send result
            result = {
                "type": "prediction",
                "timestamp": time.time(),
                "emg_data": emg_data.tolist(),
                "predictions": predictions if isinstance(predictions, list) else predictions,
                "detected_gesture": detected,
                "confidence": confidence,
                "latency_ms": latency,
                "sample_count": sample_counter
            }

            await websocket.send_json(result)
            sample_counter += 1

            # Control rate (~20Hz)
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.05 - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.websocket("/ws/file")
async def websocket_file_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming predictions from uploaded file
    """
    await websocket.accept()

    try:
        # Wait for file path message
        data = await websocket.receive_json()
        file_path = data.get("file_path")

        if not file_path or not Path(file_path).exists():
            await websocket.send_json({"error": "File not found"})
            await websocket.close()
            return

        # Load and stream file data
        import h5py
        with h5py.File(file_path, 'r') as f:
            emg_data = f['timeseries']['emg'][:]

            # Select isolated channels
            channel_indices = [4, 5, 6, 7, 8, 12, 14]
            emg_data = emg_data[:, channel_indices].T  # (7, time)

            # Stream in windows
            window_size = 100  # 50ms at 2kHz
            for i in range(0, emg_data.shape[1] - window_size, window_size):
                window = emg_data[:, i:i + window_size]

                if is_model_loaded and model is not None:
                    import torch
                    with torch.no_grad():
                        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                        output = model(input_tensor)
                        predictions = torch.sigmoid(output).squeeze().tolist()

                        max_idx = np.argmax(predictions)
                        confidence = predictions[max_idx] * 100
                        detected = max_idx if confidence > 50 else None
                else:
                    predictions, detected, confidence = generate_demo_predictions()

                result = {
                    "type": "prediction",
                    "timestamp": time.time(),
                    "emg_data": window.tolist(),
                    "predictions": predictions,
                    "detected_gesture": detected,
                    "confidence": confidence,
                    "progress": i / emg_data.shape[1]
                }

                await websocket.send_json(result)
                await asyncio.sleep(0.05)  # 20Hz playback

        await websocket.send_json({"type": "complete"})

    except WebSocketDisconnect:
        print("File stream disconnected")
    except Exception as e:
        print(f"File stream error: {e}")
        await websocket.send_json({"error": str(e)})
        await websocket.close()


@app.post("/api/upload")
async def upload_recording(file: UploadFile = File(...)):
    """Upload an EMG recording file"""
    if not file.filename.endswith('.hdf5') and not file.filename.endswith('.h5'):
        raise HTTPException(400, "Only HDF5 files are supported")

    # Save uploaded file
    upload_dir = Path(__file__).parent / "uploads"
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "filename": file.filename,
        "path": str(file_path),
        "size": len(content)
    }


@app.post("/api/predict")
async def predict_single(emg_data: list[list[float]]):
    """Single prediction from EMG data"""
    if len(emg_data) != 7:
        raise HTTPException(400, "Expected 7 channels of EMG data")

    start_time = time.time()

    if is_model_loaded and model is not None:
        import torch
        with torch.no_grad():
            input_tensor = torch.tensor(emg_data, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            predictions = torch.sigmoid(output).squeeze().tolist()

            max_idx = np.argmax(predictions)
            confidence = predictions[max_idx] * 100
            detected = max_idx if confidence > 50 else None
    else:
        predictions, detected, confidence = generate_demo_predictions()

    latency = (time.time() - start_time) * 1000

    return PredictionResult(
        timestamp=time.time(),
        emg_data=emg_data,
        predictions=predictions,
        detected_gesture=detected,
        confidence=confidence,
        latency_ms=latency
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
