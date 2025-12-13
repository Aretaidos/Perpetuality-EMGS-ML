# Perpetuality EMG - Offline Testing Tool

A beautiful, agency-quality web application for testing TinyML sEMG gesture recognition with real recorded data.

## Features

### Offline Testing
- **HDF5 Recording Browser** - Browse and select recorded EMG sessions
- **Real Model Inference** - Run your trained M1/M2 models on recorded data
- **Ground Truth Comparison** - See predictions vs actual labels side-by-side
- **Session Metrics** - Accuracy, latency, and per-gesture breakdown

### Visualization
- **Real-time Hand Visualization** - Beautiful SVG hand with glowing finger states
- **7-Channel EMG Waveform** - Live visualization of muscle signals from recordings
- **Prediction Analysis** - Probability distribution for all 9 gesture classes
- **Gesture Log** - Timestamped event history with confidence scores

### Playback Controls
- **Play/Pause** - Control recording playback
- **Speed Control** - 0.25x to 4x playback speed
- **Seek** - Jump to any point in the recording
- **Step Mode** - Frame-by-frame analysis

## Quick Start

### 1. Install Dependencies

```bash
# Frontend
cd demo-ui
npm install

# Backend (required for offline testing)
cd server
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
cd demo-ui/server
python main.py
```

This will:
- Load your trained TinyML model (M1 CNN+LSTM)
- Scan for HDF5 recordings in the data directory
- Start the WebSocket server for real-time streaming

### 3. Start the Frontend

```bash
cd demo-ui
npm run dev
```

### 4. Open in Browser

Navigate to [http://localhost:3000](http://localhost:3000)

### 5. Load a Recording

1. Click **"Select Recording"** to open the file browser
2. Choose an HDF5 recording from your data directory
3. Click **"Start Testing"** to begin playback with model inference

## What You'll See

When a recording is loaded and playing:

1. **Hand Visualization** - Fingers light up when gestures are detected
2. **Prediction Comparison** - Ground truth label vs model prediction with confidence
3. **EMG Waveform** - Real signal data from the 7 isolated channels
4. **Session Metrics** - Running accuracy and per-gesture performance
5. **Gesture Log** - All detected gestures with timestamps

## Gesture Types

| ID | Gesture | Hand Visualization |
|----|---------|-------------------|
| 0 | Index Press | Index finger glows indigo |
| 1 | Index Release | Index finger dims |
| 2 | Middle Press | Middle finger glows cyan |
| 3 | Middle Release | Middle finger dims |
| 4 | Thumb Click | Thumb pulse animation |
| 5 | Thumb Down | Thumb + down arrow |
| 6 | Thumb In | Thumb + left arrow |
| 7 | Thumb Out | Thumb + right arrow |
| 8 | Thumb Up | Thumb + up arrow |

## Architecture

```
demo-ui/
├── src/
│   ├── components/
│   │   ├── layout/           # Header, ParticleBackground
│   │   ├── hand/             # Hand visualization with finger states
│   │   ├── charts/           # EMG waveform, metrics, comparison
│   │   ├── controls/         # Recording browser, playback controls
│   │   └── log/              # Gesture event log
│   ├── types/                # TypeScript definitions
│   └── App.tsx               # Main application with WebSocket client
├── server/
│   └── main.py               # FastAPI server with model inference
└── public/
    └── assets/               # Static assets
```

## API Endpoints

### WebSocket

- `ws://localhost:8000/ws/playback` - Recording playback with inference

Commands:
```json
{"action": "load", "file_path": "/path/to/recording.hdf5"}
{"action": "play"}
{"action": "pause"}
{"action": "seek", "position": 0.5}
{"action": "speed", "value": 2.0}
{"action": "step"}
```

### REST

- `GET /api/recordings` - List all available HDF5 recordings
- `GET /api/recording/{path}` - Get recording details
- `GET /api/status` - Server status and model load state

## Data Directory

Place your HDF5 recordings in:
- `data/` directory (relative to project root)
- `recordings/` directory
- Or any subdirectory - the server scans recursively

Required HDF5 structure:
```
recording.hdf5
├── timeseries/
│   ├── emg      # (N, 16) EMG data at 2kHz
│   └── time     # (N,) timestamps
└── prompts      # Ground truth gesture labels
```

## Session Metrics

After playback completes, you'll see:

- **Overall Accuracy** - Percentage of correct predictions
- **Average Latency** - Model inference time in milliseconds
- **Average Confidence** - Mean prediction confidence
- **Per-Gesture Accuracy** - Breakdown by gesture type
- **Correct/Incorrect Count** - Visual progress bar

## Technology Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI, PyTorch, WebSockets
- **Model**: CNN+LSTM (M1) or CNN-only (M2)
- **Build**: Vite

## Design System

### Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `--accent-primary` | `#6366F1` | Index finger, primary actions |
| `--accent-secondary` | `#8B5CF6` | Thumb actions |
| `--accent-tertiary` | `#06B6D4` | Middle finger |
| `--motion-up` | `#10B981` | Thumb up indicator |
| `--motion-down` | `#F59E0B` | Thumb down indicator |

### Typography

- **Display**: Space Grotesk (metrics, large numbers)
- **Body**: Inter (UI text)
- **Mono**: JetBrains Mono (timestamps, data)

## License

MIT License - Perpetuality Neural Interfaces
