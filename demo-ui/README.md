# Perpetuality EMG - Neural Interface Demo

A beautiful, agency-quality web application for demonstrating TinyML sEMG gesture recognition.

![Demo Preview](./preview.png)

## Features

- **Real-time Hand Visualization** - Beautiful SVG hand with glowing finger states
- **7-Channel EMG Waveform** - Live visualization of muscle signals
- **Gesture Log** - Timestamped event history with confidence scores
- **Performance Metrics** - Latency, accuracy, and throughput displays
- **Offline Demo Mode** - Works without live hardware for presentations
- **Keyboard Controls** - Test gestures interactively

## Quick Start

### 1. Install Dependencies

```bash
# Frontend
cd demo-ui
npm install

# Backend (optional, for model inference)
cd server
pip install -r requirements.txt
```

### 2. Run the Demo

**Frontend Only (Demo Mode):**
```bash
npm run dev
```

**With Backend (Model Inference):**
```bash
# Terminal 1: Start backend
cd server
python main.py

# Terminal 2: Start frontend
npm run dev
```

### 3. Open in Browser

Navigate to [http://localhost:3000](http://localhost:3000)

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Start/Stop Demo |
| `I` | Toggle Index Finger Press |
| `M` | Toggle Middle Finger Press |
| `T` | Thumb Click |
| `↑` | Thumb Up |
| `↓` | Thumb Down |
| `←` | Thumb In |
| `→` | Thumb Out |

## Gesture Types

The system recognizes 9 distinct gestures:

| ID | Gesture | Description |
|----|---------|-------------|
| 0 | Index Press | Index finger pressing down |
| 1 | Index Release | Index finger releasing |
| 2 | Middle Press | Middle finger pressing down |
| 3 | Middle Release | Middle finger releasing |
| 4 | Thumb Click | Quick thumb tap |
| 5 | Thumb Down | Thumb moving downward |
| 6 | Thumb In | Thumb moving inward |
| 7 | Thumb Out | Thumb moving outward |
| 8 | Thumb Up | Thumb moving upward |

## Architecture

```
demo-ui/
├── src/
│   ├── components/
│   │   ├── layout/        # Header, Footer, Background
│   │   ├── hand/          # Hand visualization
│   │   ├── charts/        # EMG waveform, metrics
│   │   └── log/           # Gesture event log
│   ├── types/             # TypeScript definitions
│   └── App.tsx            # Main application
├── server/
│   └── main.py            # FastAPI inference server
└── public/
    └── assets/            # Static assets
```

## Technology Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI, PyTorch, WebSockets
- **Build**: Vite

## Design System

### Colors

| Token | Value | Usage |
|-------|-------|-------|
| `--accent-primary` | `#6366F1` | Index finger, primary actions |
| `--accent-secondary` | `#8B5CF6` | Thumb actions |
| `--accent-tertiary` | `#06B6D4` | Middle finger, data |
| `--finger-glow` | `rgba(99,102,241,0.4)` | Active finger glow |

### Typography

- **Display**: Space Grotesk (metrics)
- **Body**: Inter (UI text)
- **Mono**: JetBrains Mono (data, timestamps)

## API Endpoints

### WebSocket

- `ws://localhost:8000/ws/stream` - Real-time gesture streaming
- `ws://localhost:8000/ws/file` - File playback streaming

### REST

- `GET /api/status` - Server status
- `GET /api/config` - Demo configuration
- `POST /api/upload` - Upload recording file
- `POST /api/predict` - Single prediction

## Customization

### Adding New Gestures

1. Update `src/types/gestures.ts` with new gesture types
2. Add visualization in `HandVisualization.tsx`
3. Update gesture log icons in `GestureLog.tsx`

### Changing Colors

Edit `tailwind.config.js` and `src/index.css` design tokens.

## Production Build

```bash
npm run build
```

Output will be in `dist/` directory.

## License

MIT License - Perpetuality Neural Interfaces
