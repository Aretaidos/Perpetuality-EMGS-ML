# TinyML sEMG Gesture Recognition - Demo UI/UX Plan

## 🎯 Project Vision

Create a stunning, agency-quality web application for demonstrating the TinyML sEMG gesture recognition system. The UI will provide real-time visualization of gesture classifications with a futuristic, AI-inspired aesthetic.

---

## 📊 Technical Foundation

### Gesture Classes to Visualize (9 Total)
| ID | Gesture | Finger | Visual Mapping |
|----|---------|--------|----------------|
| 0 | `index_press` | Index | Finger lights up (pressed state) |
| 1 | `index_release` | Index | Finger dims (released state) |
| 2 | `middle_press` | Middle | Finger lights up |
| 3 | `middle_release` | Middle | Finger dims |
| 4 | `thumb_click` | Thumb | Quick pulse animation |
| 5 | `thumb_down` | Thumb | Downward motion indicator |
| 6 | `thumb_in` | Thumb | Inward motion indicator |
| 7 | `thumb_out` | Thumb | Outward motion indicator |
| 8 | `thumb_up` | Thumb | Upward motion indicator |

### Data Flow Architecture
```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  EMG Data Source    │────▶│  Classification  │────▶│   Web UI Display    │
│  (File/WebSocket)   │     │  Engine (Python) │     │   (React + Canvas)  │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘
         │                           │                         │
    7-channel EMG              9-class probs            Visual feedback
    @ 2000Hz                   @ 200Hz                  @ 60fps
```

---

## 🎨 Design System

### Color Palette (Futuristic AI Theme - Light Mode)

```css
/* Primary Colors */
--bg-primary: #FAFBFC;           /* Near-white background */
--bg-secondary: #F0F4F8;         /* Subtle gray panels */
--bg-glass: rgba(255,255,255,0.7); /* Glassmorphism */

/* Accent Colors (Neural/AI Theme) */
--accent-primary: #6366F1;        /* Indigo - primary actions */
--accent-secondary: #8B5CF6;      /* Purple - secondary highlights */
--accent-tertiary: #06B6D4;       /* Cyan - data indicators */

/* Finger State Colors */
--finger-idle: #E2E8F0;           /* Soft gray when inactive */
--finger-active: #6366F1;         /* Vibrant indigo when pressed */
--finger-glow: rgba(99,102,241,0.4); /* Glow effect */
--thumb-active: #8B5CF6;          /* Purple for thumb actions */
--thumb-glow: rgba(139,92,246,0.4);

/* Motion Indicators */
--motion-up: #10B981;             /* Emerald */
--motion-down: #F59E0B;           /* Amber */
--motion-in: #06B6D4;             /* Cyan */
--motion-out: #EC4899;            /* Pink */

/* Text Colors */
--text-primary: #1E293B;          /* Dark slate */
--text-secondary: #64748B;        /* Medium slate */
--text-muted: #94A3B8;            /* Light slate */

/* Status Colors */
--status-connected: #10B981;      /* Emerald green */
--status-disconnected: #EF4444;   /* Red */
--status-processing: #F59E0B;     /* Amber */
```

### Typography
- **Headings:** Inter (700, 600)
- **Body:** Inter (400, 500)
- **Monospace/Data:** JetBrains Mono
- **Display Numbers:** Space Grotesk (for large metrics)

### Visual Effects
1. **Glassmorphism** - Frosted glass panels with backdrop blur
2. **Subtle Gradients** - Mesh gradient backgrounds
3. **Soft Shadows** - Layered, diffused shadows
4. **Glow Effects** - Finger activation states
5. **Smooth Animations** - 60fps transitions with easing
6. **Particle Effects** - Subtle floating particles for atmosphere

---

## 🖐️ Hand Visualization Design

### SVG Hand Component
```
                    ┌─────────────────────────────────────────┐
                    │           PERPETUALITY EMG              │
                    │        Neural Interface Demo            │
                    ├─────────────────────────────────────────┤
                    │                                         │
                    │         ┌───┐                          │
                    │         │ M │  ← Middle Finger         │
                    │     ┌───┼───┼───┐                      │
                    │     │ I │   │ R │  ← Ring (future)     │
                    │ ┌───┼───┼───┼───┼───┐                  │
                    │ │ T │   │   │   │ P │  ← Pinky (future)│
                    │ └───┴───┴───┴───┴───┘                  │
                    │     └── Index                           │
                    │  Thumb ──┘                              │
                    │                                         │
                    │  ═══════════════════════════════════   │
                    │  EMG Signal Waveform Visualization      │
                    │  ═══════════════════════════════════   │
                    │                                         │
                    └─────────────────────────────────────────┘
```

### Finger States & Animations
| State | Visual Treatment |
|-------|------------------|
| **Idle** | Soft gray fill, subtle inner shadow |
| **Press** | Indigo fill with outer glow, scale 1.05 |
| **Release** | Quick fade from active to idle |
| **Thumb Click** | Pulse animation (scale + glow burst) |
| **Thumb Direction** | Arrow indicator + directional gradient |

---

## 📐 UI Layout Structure

### Main Dashboard (1440px design)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HEADER                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 🧠 PERPETUALITY  │  EMG Neural Interface  │  ● Connected  │  ⚙️ Settings ││
│  └─────────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────────────────────┐  ┌────────────────────────────────┐ │
│   │                                   │  │   GESTURE LOG                  │ │
│   │                                   │  │   ─────────────────────────    │ │
│   │         HAND VISUALIZATION        │  │   12:34:56  Index Press   ●   │ │
│   │                                   │  │   12:34:57  Index Release ○   │ │
│   │      [Animated SVG Hand with      │  │   12:34:58  Thumb Click   ◆   │ │
│   │       glowing finger states]      │  │   12:34:59  Middle Press  ●   │ │
│   │                                   │  │   12:35:01  Thumb Up      ↑   │ │
│   │                                   │  │   ...                          │ │
│   └───────────────────────────────────┘  └────────────────────────────────┘ │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  EMG SIGNAL MONITOR  (7-Channel Real-time Waveform)                 │   │
│   │  ════════════════════════════════════════════════════════════════   │   │
│   │  Ch5 ──────────────────────────────────────────────────────────     │   │
│   │  Ch6 ──────────────────────────────────────────────────────────     │   │
│   │  Ch7 ──────────────────────────────────────────────────────────     │   │
│   │  Ch8 ──────────────────────────────────────────────────────────     │   │
│   │  Ch9 ──────────────────────────────────────────────────────────     │   │
│   │  Ch13 ─────────────────────────────────────────────────────────     │   │
│   │  Ch15 ─────────────────────────────────────────────────────────     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│   │  MODEL CONFIDENCE  │  │   LATENCY          │  │   ACCURACY         │   │
│   │       94.7%        │  │      4.2ms         │  │     96.3%          │   │
│   │  ████████████░░░░  │  │  ██░░░░░░░░░░░░░░  │  │  █████████████░░░  │   │
│   └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  FOOTER                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ▶ Start Demo  │  📂 Load Recording  │  🔄 Reset  │  📊 Export Report  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Implementation

### Technology Stack
| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React 18 + TypeScript | Component architecture |
| **Styling** | Tailwind CSS + Custom CSS | Utility-first with design tokens |
| **Animation** | Framer Motion | Smooth gesture animations |
| **Charts** | Chart.js / Recharts | EMG waveform visualization |
| **3D (optional)** | Three.js | Enhanced hand model |
| **Backend** | FastAPI (Python) | Model inference server |
| **WebSocket** | Socket.io | Real-time data streaming |
| **Build** | Vite | Fast development/bundling |

### File Structure
```
demo-ui/
├── public/
│   ├── index.html
│   └── assets/
│       ├── hand-outline.svg
│       └── perpetuality-logo.svg
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── Dashboard.tsx
│   │   ├── hand/
│   │   │   ├── HandVisualization.tsx
│   │   │   ├── Finger.tsx
│   │   │   ├── ThumbIndicator.tsx
│   │   │   └── GlowEffect.tsx
│   │   ├── charts/
│   │   │   ├── EMGWaveform.tsx
│   │   │   ├── ConfidenceGauge.tsx
│   │   │   └── MetricsCard.tsx
│   │   ├── controls/
│   │   │   ├── DemoControls.tsx
│   │   │   ├── FileUploader.tsx
│   │   │   └── SettingsPanel.tsx
│   │   └── log/
│   │       ├── GestureLog.tsx
│   │       └── GestureLogItem.tsx
│   ├── hooks/
│   │   ├── useGestureStream.ts
│   │   ├── useEMGData.ts
│   │   └── useAnimation.ts
│   ├── services/
│   │   ├── websocket.ts
│   │   ├── classifier.ts
│   │   └── dataLoader.ts
│   ├── types/
│   │   └── gestures.ts
│   ├── utils/
│   │   ├── colors.ts
│   │   └── animations.ts
│   └── styles/
│       ├── design-tokens.css
│       └── animations.css
├── server/
│   ├── main.py              # FastAPI server
│   ├── inference.py         # Model inference wrapper
│   └── requirements.txt
├── package.json
├── tailwind.config.js
├── vite.config.ts
└── tsconfig.json
```

---

## 🔄 Data Flow Implementation

### 1. Offline Demo Mode (File Playback)
```python
# Server-side: Stream pre-recorded data
async def stream_recording(file_path: str):
    recording = EmgRecording(Path(file_path))
    for window in sliding_window(recording):
        predictions = model.predict(window)
        yield {
            "timestamp": time.time(),
            "emg_data": window.tolist(),
            "predictions": predictions.tolist(),
            "confidences": torch.sigmoid(predictions).tolist()
        }
        await asyncio.sleep(0.005)  # 200Hz output
```

### 2. Real-time Demo Mode (Live Sensor)
```python
# Server-side: Process live EMG stream
async def process_live_stream():
    async for data in emg_sensor.stream():
        predictions = model.predict(data)
        await websocket.send_json({
            "type": "prediction",
            "data": predictions.tolist()
        })
```

### 3. Frontend State Management
```typescript
interface GestureState {
  index: { pressed: boolean; confidence: number };
  middle: { pressed: boolean; confidence: number };
  thumb: {
    action: 'idle' | 'click' | 'up' | 'down' | 'in' | 'out';
    confidence: number;
  };
  emgChannels: number[][];
  gestureLog: GestureEvent[];
  metrics: {
    latency: number;
    accuracy: number;
    modelConfidence: number;
  };
}
```

---

## ✨ Animation Specifications

### Finger Press Animation
```css
@keyframes fingerPress {
  0% {
    fill: var(--finger-idle);
    filter: drop-shadow(0 0 0 transparent);
    transform: scale(1);
  }
  50% {
    fill: var(--finger-active);
    filter: drop-shadow(0 0 20px var(--finger-glow));
    transform: scale(1.08);
  }
  100% {
    fill: var(--finger-active);
    filter: drop-shadow(0 0 12px var(--finger-glow));
    transform: scale(1.05);
  }
}
```

### Thumb Click Pulse
```css
@keyframes thumbPulse {
  0% {
    fill: var(--thumb-active);
    filter: drop-shadow(0 0 0 transparent);
    transform: scale(1);
  }
  25% {
    fill: var(--thumb-active);
    filter: drop-shadow(0 0 30px var(--thumb-glow));
    transform: scale(1.15);
  }
  100% {
    fill: var(--finger-idle);
    filter: drop-shadow(0 0 0 transparent);
    transform: scale(1);
  }
}
```

### Directional Motion Indicator
```css
@keyframes motionUp {
  0%, 100% { transform: translateY(0); opacity: 0.3; }
  50% { transform: translateY(-8px); opacity: 1; }
}

@keyframes motionDown {
  0%, 100% { transform: translateY(0); opacity: 0.3; }
  50% { transform: translateY(8px); opacity: 1; }
}
```

---

## 📱 Responsive Breakpoints

| Breakpoint | Width | Layout Adjustments |
|------------|-------|-------------------|
| **Desktop** | ≥1280px | Full dashboard with side panel |
| **Laptop** | ≥1024px | Condensed metrics row |
| **Tablet** | ≥768px | Stacked layout, smaller hand |
| **Mobile** | <768px | Single column, scrollable |

---

## 🚀 Implementation Phases

### Phase 1: Foundation (Core UI)
- [ ] Project setup (Vite + React + TypeScript + Tailwind)
- [ ] Design system implementation (colors, typography, tokens)
- [ ] Basic layout components (Header, Footer, Dashboard)
- [ ] Static hand SVG visualization
- [ ] Finger components with CSS animations

### Phase 2: Interactivity
- [ ] Gesture state management
- [ ] Finger press/release animations
- [ ] Thumb action indicators
- [ ] Glow effects and transitions
- [ ] Keyboard shortcuts for demo testing

### Phase 3: Data Visualization
- [ ] EMG waveform chart component
- [ ] Real-time streaming visualization
- [ ] Confidence gauges
- [ ] Metrics cards
- [ ] Gesture event log

### Phase 4: Backend Integration
- [ ] FastAPI inference server
- [ ] WebSocket connection
- [ ] File upload for recordings
- [ ] Model loading and inference
- [ ] Data streaming pipeline

### Phase 5: Polish & Demo Mode
- [ ] Particle background effects
- [ ] Loading states and transitions
- [ ] Error handling UI
- [ ] Demo playback controls
- [ ] Export functionality

---

## 🎭 Demo Scenarios

### Scenario 1: Live Presentation
1. Present with device on presenter's arm
2. Show real-time finger tracking
3. Demonstrate each gesture type
4. Show latency and accuracy metrics

### Scenario 2: Recorded Playback
1. Load pre-recorded session
2. Playback with speed controls
3. Pause and analyze specific moments
4. Export highlight clips

### Scenario 3: Interactive Testing
1. Keyboard simulation mode
2. Test all gesture animations
3. Verify visual feedback
4. Stress test with rapid inputs

---

## 🔐 Offline Capability

The demo UI will work fully offline:
- All assets bundled locally
- Model runs on local Python server
- No external API dependencies
- Can use pre-recorded data files

---

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| **First Contentful Paint** | <1.5s |
| **Animation Frame Rate** | 60fps |
| **Gesture Update Latency** | <50ms |
| **Bundle Size** | <500KB gzipped |
| **Lighthouse Score** | >90 |

---

## 🎨 Design Inspiration References

1. **Linear** - Clean, minimal interface with subtle animations
2. **Vercel** - Dark/light mode with gradient accents
3. **Stripe** - Polished gradients and micro-interactions
4. **Apple Health** - Data visualization excellence
5. **Figma** - Real-time collaboration indicators
6. **Raycast** - Keyboard-first, snappy interactions

---

## Next Steps

1. ✅ Research and planning complete
2. 🔄 Create project structure
3. 🔄 Implement design system
4. 🔄 Build hand visualization component
5. 🔄 Add animation system
6. 🔄 Integrate with inference backend
7. 🔄 Polish and optimize

---

*Last Updated: December 2024*
*Version: 1.0.0*
