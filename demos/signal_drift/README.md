# Signal Drift - EMG Gesture Runner Demo

A side-scrolling runner game controlled by EMG signals that demonstrates realistic gesture recognition capabilities with the M1 4-channel TinyML model.

## Concept

**Signal Drift** turns the limitations of the 4-channel EMG model (~36% gesture recall) into gameplay features:

- **Primary Control**: Raw EMG amplitude controls vertical position (reliable, no classification needed)
- **Bonus Power-Ups**: Confident gesture detections (>70%) trigger special abilities
- **Live Visualization**: Shows all 9 gesture confidences as they change in real-time
- **Educational**: Demonstrates how ML classification uncertainty works

## Features

### Gameplay
- Control a glowing orb through a scrolling tunnel
- Dodge obstacles, collect coins
- Trigger power-ups with confident gesture detections
- Score points for distance + gesture streaks

### 9 Gesture Power-Ups
When the model detects a gesture with >70% confidence:

| Gesture | Power-Up | Effect | Duration |
|---------|----------|--------|----------|
| index_press | Speed Boost | 1.5x scroll speed | 3s |
| index_release | Slow Motion | 0.5x scroll speed | 4s |
| middle_press | Shield | Invincible to obstacles | 5s |
| middle_release | Double Points | 2x coin value | 5s |
| thumb_click | Magnet | Auto-collect coins | 4s |
| thumb_down | Gravity Flip | Inverted controls | 3s |
| thumb_in | Shrink | Smaller orb, easier dodging | 4s |
| thumb_out | Grow | Larger orb, destroys obstacles | 3s |
| thumb_up | Flight Mode | Enhanced lift force | 4s |

### Visual Feedback
- **Live confidence display**: 9 rows showing real-time ML output
- **Detection badges**: "✓ DETECTED!" when gesture confidence >70%
- **Color-coded bars**: Green for detected, gray for inactive
- **EMG amplitude meter**: Shows raw signal strength
- **Power-up timers**: Active effects with countdown

## Installation

```bash
cd demos/signal_drift
pip install -r requirements.txt
```

## Usage

### Keyboard Mode (Default)
```bash
python main.py
```

**Controls:**
- `UP/DOWN`: Simulate EMG signal strength
- `1-9`: Trigger gesture detections (1=index_press, 2=index_release, ... 9=thumb_up)
- `M`: Toggle hardware/keyboard mode
- `P`: Pause
- `ESC`: Quit

### Hardware Mode (XIAO nRF52840)
```bash
python main.py --hardware
```

**Requirements:**
- XIAO nRF52840 connected via USB
- M1 4-channel TinyML firmware running
- Serial port: `/dev/ttyACM0` @ 115200 baud

**How it works:**
- Reads JSON from serial: `{"gesture": "thumb_click", "conf": 0.92, "scores": [...]}`
- Any muscle activation → orb rises
- Relaxed state → orb falls
- Confident detections → trigger power-ups

## Architecture

```
signal_drift/
├── main.py                 # Game loop, rendering, collision detection
├── gesture_input.py        # Serial hardware + keyboard simulation
├── config.py               # Constants, gesture mappings, colors
├── requirements.txt        # Dependencies
└── README.md               # This file
```

### Key Classes

**`GestureInput`** - Dual-mode input handler
- Hardware mode: Reads JSON from serial
- Keyboard mode: Simulates EMG signals
- Auto-fallback if hardware disconnects

**`KeyboardGestureSimulator`** - Test gesture detections without hardware
- Trigger gestures via number keys
- Decay confidence over time
- Useful for development/demo prep

**`SignalDriftGame`** - Main Arcade window
- Updates at 60 FPS
- Physics: EMG amplitude → lift force
- Collision detection with obstacles/coins
- Power-up management

## Design Rationale

### Why This Approach?

The M1 4-channel model has **~36% recall** for all 9 gestures (based on evaluation data). Traditional UIs requiring precise gesture control would be frustrating. Signal Drift solves this:

1. **Primary control doesn't need classification** - uses raw EMG amplitude (very reliable)
2. **Gesture detection is bonus** - power-ups are exciting when they trigger, not required to play
3. **Low recall becomes rarity** - makes power-ups feel special, not common
4. **Shows ML honestly** - confidence display educates about model uncertainty
5. **Forgiving gameplay** - missing detections doesn't = instant failure

### Performance Expectations

With hardware (4-channel XIAO):
- **EMG control**: Works consistently (raw amplitude is stable)
- **Gesture power-ups**: Trigger ~1-2 times per minute with normal hand movements
- **Confident detections (>70%)**: Rare but achievable with deliberate gestures
- **False positives**: Beneficial (power-ups help, don't hurt)

## Troubleshooting

**"Cannot connect to hardware"**
- Check USB connection: `ls /dev/ttyACM*`
- Verify firmware is running: Send 's' command via serial monitor
- Try different port: Edit `SERIAL_PORT` in `config.py`

**"Gestures not detecting"**
- Check firmware output: Confidence scores in JSON
- Lower threshold: Press `[` key to reduce from 70% to 60%
- Verify electrode placement: Forearm flexor/extensor muscles
- Hardware may auto-switch to keyboard mode if serial fails

**"Game too fast/slow"**
- Edit `TUNNEL_SCROLL_SPEED` in `config.py`
- Adjust `ORB_GRAVITY` and `ORB_LIFT_FORCE` for different feel

## Future Enhancements

- [ ] Add tutorial mode explaining each gesture
- [ ] Record/replay EMG sequences for reproducible demos
- [ ] Adaptive difficulty based on gesture detection success rate
- [ ] Sound effects and background music
- [ ] Leaderboard with gesture streak tracking
- [ ] Export gameplay video with EMG overlay

## Credits

Built for demonstrating the M1 4-Channel TinyML sEMG gesture recognition model.

Hardware: Seeed Studio XIAO nRF52840
Model: CNN+LSTM (4 channels, 9 gesture classes, 99.6% validation accuracy)
Framework: Python Arcade
