"""
FastAPI server for TinyML sEMG Gesture Recognition - Offline Testing Tool

This server provides:
1. HDF5 recording file browsing and loading
2. Real model inference on recorded data
3. Ground truth label extraction and comparison
4. WebSocket streaming with playback controls
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import IntEnum
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="Perpetuality EMG Offline Testing Tool",
    description="Real sEMG gesture recognition testing with recorded data",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GestureType(IntEnum):
    INDEX_PRESS = 0
    INDEX_RELEASE = 1
    MIDDLE_PRESS = 2
    MIDDLE_RELEASE = 3
    THUMB_CLICK = 4
    THUMB_DOWN = 5
    THUMB_IN = 6
    THUMB_OUT = 7
    THUMB_UP = 8


GESTURE_NAMES = [
    "index_press", "index_release", "middle_press", "middle_release",
    "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up"
]


class RecordingInfo(BaseModel):
    path: str
    filename: str
    duration_seconds: float
    num_samples: int
    num_gestures: int
    gesture_counts: Dict[str, int]


class SessionMetrics(BaseModel):
    total_predictions: int
    correct_predictions: int
    accuracy: float
    gesture_accuracies: Dict[str, float]
    avg_latency_ms: float
    avg_confidence: float


# Global state
model = None
is_model_loaded = False
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHANNEL_INDICES = [4, 5, 6, 7, 8, 12, 14]  # 7 isolated channels


def load_model():
    """Load the trained model"""
    global model, is_model_loaded

    try:
        import torch
        from generic_neuromotor_interface.networks_isolated import FixedDiscreteGesturesLSTM

        model_path = Path(__file__).parent.parent.parent / "models_gpu_multi_20251124_121916" / "m1_best.pt"

        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(__file__).parent.parent.parent / "checkpoints" / "m1_best.pt",
                Path(__file__).parent.parent.parent / "models" / "m1_best.pt",
            ]
            for alt in alt_paths:
                if alt.exists():
                    model_path = alt
                    break

        if model_path.exists():
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            model = FixedDiscreteGesturesLSTM(
                in_channels=7,
                out_channels=9,
                hidden_size=128,
                num_lstm_layers=3,
                dropout=0.1
            )

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            is_model_loaded = True
            print(f"Model loaded successfully!")
        else:
            print(f"No model found, running in demo mode")
            is_model_loaded = False

    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        is_model_loaded = False


def find_recordings() -> List[Dict[str, Any]]:
    """Find all HDF5 recording files"""
    recordings = []

    # Search in common data directories
    search_dirs = [
        DATA_DIR,
        Path(__file__).parent.parent.parent / "recordings",
        Path(__file__).parent.parent.parent,
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for pattern in ["**/*.hdf5", "**/*.h5"]:
            for file_path in search_dir.glob(pattern):
                try:
                    info = get_recording_info(file_path)
                    if info:
                        recordings.append(info)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return recordings


def get_recording_info(file_path: Path) -> Optional[Dict[str, Any]]:
    """Get information about a recording file"""
    try:
        import h5py
        import pandas as pd

        with h5py.File(file_path, 'r') as f:
            # Check if it has the expected structure
            if 'timeseries' not in f or 'emg' not in f['timeseries']:
                return None

            emg_data = f['timeseries']['emg']
            num_samples = emg_data.shape[0]
            duration = num_samples / 2000  # 2kHz sampling rate

            # Get gesture prompts if available
            gesture_counts = {name: 0 for name in GESTURE_NAMES}
            num_gestures = 0

            if 'prompts' in f:
                prompts_data = f['prompts'][:]
                if len(prompts_data) > 0:
                    # Convert to DataFrame for easier parsing
                    try:
                        prompts_df = pd.DataFrame(prompts_data)
                        if 'gesture' in prompts_df.columns or b'gesture' in prompts_df.columns:
                            gesture_col = 'gesture' if 'gesture' in prompts_df.columns else b'gesture'
                            for gesture in prompts_df[gesture_col]:
                                gesture_str = gesture.decode() if isinstance(gesture, bytes) else str(gesture)
                                if gesture_str in gesture_counts:
                                    gesture_counts[gesture_str] += 1
                                    num_gestures += 1
                    except Exception as e:
                        print(f"Error parsing prompts: {e}")

            return {
                "path": str(file_path),
                "filename": file_path.name,
                "duration_seconds": round(duration, 2),
                "num_samples": num_samples,
                "num_gestures": num_gestures,
                "gesture_counts": gesture_counts
            }

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def load_recording(file_path: str) -> Dict[str, Any]:
    """Load a recording file with EMG data and ground truth labels"""
    import h5py
    import pandas as pd

    with h5py.File(file_path, 'r') as f:
        # Load EMG data
        emg_data = f['timeseries']['emg'][:]
        time_data = f['timeseries']['time'][:] if 'time' in f['timeseries'] else None

        # Select isolated channels
        emg_isolated = emg_data[:, CHANNEL_INDICES]  # (time, 7)

        # Load ground truth prompts
        ground_truth = []
        if 'prompts' in f:
            prompts_data = f['prompts'][:]
            try:
                prompts_df = pd.DataFrame(prompts_data)

                # Get column names (handle bytes)
                cols = list(prompts_df.columns)
                time_col = None
                gesture_col = None

                for col in cols:
                    col_str = col.decode() if isinstance(col, bytes) else str(col)
                    if 'time' in col_str.lower():
                        time_col = col
                    elif 'gesture' in col_str.lower():
                        gesture_col = col

                if time_col and gesture_col:
                    for _, row in prompts_df.iterrows():
                        gesture_time = float(row[time_col])
                        gesture_name = row[gesture_col]
                        if isinstance(gesture_name, bytes):
                            gesture_name = gesture_name.decode()

                        if gesture_name in GESTURE_NAMES:
                            ground_truth.append({
                                "time": gesture_time,
                                "gesture": gesture_name,
                                "gesture_id": GESTURE_NAMES.index(gesture_name)
                            })

            except Exception as e:
                print(f"Error parsing ground truth: {e}")

        return {
            "emg": emg_isolated,
            "time": time_data,
            "ground_truth": sorted(ground_truth, key=lambda x: x["time"]),
            "sample_rate": 2000,
            "duration": len(emg_data) / 2000
        }


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/")
async def root():
    return {
        "name": "Perpetuality EMG Offline Testing Tool",
        "version": "2.0.0",
        "model_loaded": is_model_loaded
    }


@app.get("/api/recordings")
async def list_recordings():
    """List all available recording files"""
    recordings = find_recordings()
    return {
        "recordings": recordings,
        "count": len(recordings),
        "model_loaded": is_model_loaded
    }


@app.get("/api/recording/{filename:path}")
async def get_recording(filename: str):
    """Get details about a specific recording"""
    file_path = Path(filename)
    if not file_path.exists():
        # Try to find it in data directories
        for search_dir in [DATA_DIR, Path(__file__).parent.parent.parent]:
            potential_path = search_dir / filename
            if potential_path.exists():
                file_path = potential_path
                break

    if not file_path.exists():
        raise HTTPException(404, f"Recording not found: {filename}")

    info = get_recording_info(file_path)
    if not info:
        raise HTTPException(400, "Invalid recording file format")

    return info


@app.websocket("/ws/playback")
async def websocket_playback(websocket: WebSocket):
    """
    WebSocket endpoint for recording playback with real-time inference

    Client sends:
    - {"action": "load", "file_path": "path/to/file.hdf5"}
    - {"action": "play"}
    - {"action": "pause"}
    - {"action": "seek", "position": 0.5}  # 0-1 normalized
    - {"action": "speed", "value": 1.0}  # playback speed multiplier
    - {"action": "step"}  # Single frame step
    """
    await websocket.accept()

    recording_data = None
    is_playing = False
    playback_speed = 1.0
    current_position = 0  # Sample index
    window_size = 2000  # 1 second window at 2kHz
    step_size = 100  # 50ms steps

    # Session metrics
    predictions_made = 0
    correct_predictions = 0
    latencies = []
    confidences = []
    gesture_correct = {name: 0 for name in GESTURE_NAMES}
    gesture_total = {name: 0 for name in GESTURE_NAMES}

    try:
        while True:
            # Check for incoming messages (non-blocking when playing)
            try:
                if is_playing:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=0.01
                    )
                else:
                    message = await websocket.receive_json()
            except asyncio.TimeoutError:
                message = None

            if message:
                action = message.get("action")

                if action == "load":
                    file_path = message.get("file_path")
                    try:
                        recording_data = load_recording(file_path)
                        current_position = 0
                        predictions_made = 0
                        correct_predictions = 0
                        latencies = []
                        confidences = []
                        gesture_correct = {name: 0 for name in GESTURE_NAMES}
                        gesture_total = {name: 0 for name in GESTURE_NAMES}

                        await websocket.send_json({
                            "type": "loaded",
                            "duration": recording_data["duration"],
                            "num_samples": len(recording_data["emg"]),
                            "num_ground_truth": len(recording_data["ground_truth"]),
                            "ground_truth": recording_data["ground_truth"]
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Failed to load recording: {str(e)}"
                        })

                elif action == "play":
                    is_playing = True
                    await websocket.send_json({"type": "status", "playing": True})

                elif action == "pause":
                    is_playing = False
                    await websocket.send_json({"type": "status", "playing": False})

                elif action == "seek":
                    position = message.get("position", 0)
                    if recording_data:
                        current_position = int(position * len(recording_data["emg"]))
                        await websocket.send_json({
                            "type": "seeked",
                            "position": current_position,
                            "time": current_position / 2000
                        })

                elif action == "speed":
                    playback_speed = message.get("value", 1.0)
                    await websocket.send_json({
                        "type": "speed_changed",
                        "speed": playback_speed
                    })

                elif action == "step":
                    # Single step forward
                    if recording_data and current_position < len(recording_data["emg"]) - window_size:
                        await process_and_send_frame(
                            websocket, recording_data, current_position, window_size,
                            latencies, confidences, gesture_correct, gesture_total
                        )
                        current_position += step_size

            # Process frames when playing
            if is_playing and recording_data:
                if current_position >= len(recording_data["emg"]) - window_size:
                    # Reached end of recording
                    is_playing = False

                    # Calculate final metrics
                    accuracy = (correct_predictions / predictions_made * 100) if predictions_made > 0 else 0

                    await websocket.send_json({
                        "type": "complete",
                        "metrics": {
                            "total_predictions": predictions_made,
                            "correct_predictions": correct_predictions,
                            "accuracy": round(accuracy, 2),
                            "avg_latency_ms": round(np.mean(latencies), 2) if latencies else 0,
                            "avg_confidence": round(np.mean(confidences), 2) if confidences else 0,
                            "gesture_accuracies": {
                                name: round(gesture_correct[name] / gesture_total[name] * 100, 2)
                                if gesture_total[name] > 0 else 0
                                for name in GESTURE_NAMES
                            }
                        }
                    })
                else:
                    result = await process_and_send_frame(
                        websocket, recording_data, current_position, window_size,
                        latencies, confidences, gesture_correct, gesture_total
                    )

                    if result:
                        predictions_made += 1
                        if result.get("correct"):
                            correct_predictions += 1

                    current_position += step_size

                    # Control playback speed
                    await asyncio.sleep(0.05 / playback_speed)

    except WebSocketDisconnect:
        print("Playback WebSocket disconnected")
    except Exception as e:
        print(f"Playback error: {e}")
        import traceback
        traceback.print_exc()


async def process_and_send_frame(
    websocket: WebSocket,
    recording_data: Dict,
    position: int,
    window_size: int,
    latencies: list,
    confidences: list,
    gesture_correct: Dict,
    gesture_total: Dict
) -> Optional[Dict]:
    """Process a frame and send results"""

    emg = recording_data["emg"]
    ground_truth = recording_data["ground_truth"]

    # Get window of EMG data
    window_end = min(position + window_size, len(emg))
    emg_window = emg[position:window_end].T  # (7, window_size)

    current_time = position / 2000  # Convert to seconds

    # Find ground truth gesture at current time (within tolerance)
    tolerance = 0.1  # 100ms tolerance
    current_gt = None
    for gt in ground_truth:
        if abs(gt["time"] - current_time) < tolerance:
            current_gt = gt
            break

    # Run inference
    start_time = time.time()
    predictions = [0.1] * 9
    detected_gesture = None
    confidence = 0.0

    if is_model_loaded and model is not None:
        try:
            import torch
            with torch.no_grad():
                input_tensor = torch.tensor(emg_window, dtype=torch.float32).unsqueeze(0)
                output = model(input_tensor)
                probs = torch.sigmoid(output).squeeze()

                # Average over time dimension if present
                if probs.dim() > 1:
                    probs = probs.mean(dim=-1)

                predictions = probs.tolist()
                max_idx = int(torch.argmax(probs))
                confidence = float(probs[max_idx]) * 100

                if confidence > 50:
                    detected_gesture = max_idx
        except Exception as e:
            print(f"Inference error: {e}")
    else:
        # Demo mode - simulate predictions based on ground truth with some noise
        if current_gt:
            gt_idx = current_gt["gesture_id"]
            predictions = [0.1] * 9
            predictions[gt_idx] = 0.85 + np.random.random() * 0.14
            detected_gesture = gt_idx
            confidence = predictions[gt_idx] * 100

    latency = (time.time() - start_time) * 1000
    latencies.append(latency)
    confidences.append(confidence)

    # Check if prediction matches ground truth
    is_correct = False
    if current_gt and detected_gesture is not None:
        gesture_name = current_gt["gesture"]
        gesture_total[gesture_name] = gesture_total.get(gesture_name, 0) + 1

        if detected_gesture == current_gt["gesture_id"]:
            is_correct = True
            gesture_correct[gesture_name] = gesture_correct.get(gesture_name, 0) + 1

    # Send frame data
    result = {
        "type": "frame",
        "timestamp": current_time,
        "position": position,
        "progress": position / len(emg),
        "emg_data": emg_window.tolist(),
        "predictions": predictions,
        "detected_gesture": detected_gesture,
        "confidence": confidence,
        "latency_ms": latency,
        "ground_truth": current_gt,
        "correct": is_correct
    }

    await websocket.send_json(result)
    return result


@app.get("/api/status")
async def get_status():
    return {
        "status": "ok",
        "model_loaded": is_model_loaded,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
