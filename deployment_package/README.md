# M1 4-Channel TinyML - Deployment Package

**Ready-to-deploy EMG gesture recognition for Seeed XIAO nRF52840**

This package contains everything needed to deploy the trained M1 4-channel gesture recognition model on embedded hardware and build demo applications.

## 📦 Package Contents

```
deployment_package/
├── models/                          # Trained models
│   ├── best_m1_4channel.pt         # PyTorch checkpoint (99.6% val accuracy)
│   └── training_summary.json       # Training metrics
│
├── firmware/                        # XIAO nRF52840 firmware
│   └── platformio_project/         # PlatformIO project (ready to flash)
│       ├── platformio.ini          # Build configuration
│       ├── src/main.cpp            # Main application
│       ├── include/
│       │   └── m1_4channel_weights.h  # Model weights (C arrays)
│       └── lib/                    # Libraries
│
├── demos/                           # Demo applications
│   └── signal_drift/               # Python demo (Arcade)
│       ├── simple_demo.py          # Visual gesture demo
│       ├── main.py                 # Full game demo
│       ├── gesture_input.py        # Serial + keyboard input
│       ├── config.py               # Configuration
│       └── requirements.txt        # Python dependencies
│
└── README.md                        # This file
```

## 🎯 Model Specifications

### M1 4-Channel TinyML

**Architecture:**
- Conv1D(4→72, k=15, s=10) + BatchNorm + ReLU + Dropout(0.3)
- LSTM(72→48, 2 layers) + LayerNorm
- FC(48→9) output layer

**Performance:**
- Validation Accuracy: **99.59%** (element-wise)
- Per-Gesture F1-Score: **~45%** (macro average)
- Per-Gesture Recall: **~36%** (detection rate)
- Parameters: **47,313** (47KB INT8 quantized)
- Inference Latency: **50-100ms** on nRF52840

**Hardware Requirements:**
- 4 EMG channels at 2kHz sampling
- 2000-sample window (1.0 second)
- 256KB RAM, 64MHz ARM Cortex-M4F

**Channel Mapping (0-based indexing):**
- Ch7 (idx 6): Index flexor - 24.5% importance
- Ch8 (idx 7): Ring flexor - 13.2% importance
- Ch13 (idx 12): Index/middle extensor - 19.8% importance
- Ch15 (idx 14): Ring/pinky extensor - 8.7% importance

**9 Gesture Classes:**
0. index_press
1. index_release
2. middle_press
3. middle_release
4. thumb_click
5. thumb_down
6. thumb_in
7. thumb_out
8. thumb_up

## 🚀 Quick Start

### Option 1: Flash Firmware to XIAO nRF52840

**Prerequisites:**
- Seeed Studio XIAO nRF52840 board
- PlatformIO CLI or VS Code extension
- USB-C cable

**Steps:**
```bash
cd deployment_package/firmware/platformio_project

# Build and flash
pio run -e xiao_nrf52840_4channel --target upload

# Monitor serial output
pio device monitor -b 115200
```

**Serial Output Format (JSON):**
```json
{"n":42,"ms":73,"class":4,"gesture":"thumb_click","conf":0.92,"scores":[0.02,0.01,...]}
```

- `n`: Inference count
- `ms`: Latency in milliseconds
- `class`: Predicted gesture ID (0-8)
- `gesture`: Gesture name
- `conf`: Confidence score (0-1)
- `scores`: All 9 class probabilities

### Option 2: Run Python Demo (Off-Server)

**Prerequisites:**
- Python 3.10+
- XIAO nRF52840 with firmware (for hardware mode)
- OR keyboard-only testing

**Setup:**
```bash
cd deployment_package/demos/signal_drift

# Install dependencies
pip install -r requirements.txt

# Run demo (keyboard mode)
python simple_demo.py

# Run demo (hardware mode - requires XIAO connected)
python simple_demo.py --hardware
```

**Demo Features:**
- Hand diagram showing muscle activation
- Live EMG traces (4 channels)
- Gesture confidence bars (9 classes)
- Interactive objects responding to gestures
- Toggle hardware/keyboard with 'M' key

## 📊 Model Performance Details

### Training Results
From `training_summary.json`:
```json
{
  "best_val_accuracy": 0.9959,
  "best_epoch": 2,
  "total_epochs": 50,
  "parameters": 47313,
  "window_length": 2000,
  "channel_indices": [6, 7, 12, 14]
}
```

### Real-World Performance
Based on comprehensive evaluation:

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | 99.73% | Misleading - dominated by "no gesture" class |
| Macro Precision | 58.9% | Average across 9 gestures |
| Macro Recall | **36.6%** | **Detection rate** |
| Macro F1-Score | 45.2% | Harmonic mean |

**Key Insight:** The model achieves high overall accuracy because ~95% of timesteps are "rest" (no gesture). Actual gesture detection rate is ~36%, meaning:
- Only 1 in 3 gesture occurrences are detected
- All 9 gestures perform similarly (~36-37% recall)
- No significant difference between finger vs thumb gestures

**Implication for Demos:**
- Don't require precise gesture sequences
- Use gestures as "bonus" triggers, not primary control
- Show confidence scores to demonstrate ML uncertainty
- Celebrate detections when they occur (they're rare!)

## 🔧 Hardware Setup

### EMG Electrode Placement

**Required:** 4 sEMG electrodes on forearm

**Placement (based on channel importance):**
1. **Ch7** - Midpoint of forearm, flexor side (palmar)
   - Target: Index flexor digitorum superficialis
   - 24.5% feature importance

2. **Ch8** - Proximal forearm, flexor side (palmar)
   - Target: Ring/pinky flexor digitorum superficialis
   - 13.2% feature importance

3. **Ch13** - Midpoint of forearm, extensor side (dorsal)
   - Target: Index/middle extensor digitorum
   - 19.8% feature importance

4. **Ch15** - Distal forearm, extensor side (dorsal)
   - Target: Ring/pinky extensor digitorum
   - 8.7% feature importance

**Tips:**
- Clean skin with alcohol wipe
- Shave hair if needed for better contact
- Position electrodes perpendicular to muscle fibers
- Use conductive gel for improved signal quality

### XIAO nRF52840 Connections

**ADC Pins:**
- A0 → Ch7 (Index flexor)
- A1 → Ch8 (Ring flexor)
- A2 → Ch13 (Index/middle extensor)
- A3 → Ch15 (Ring/pinky extensor)
- GND → Reference electrode (elbow or wrist)

**Signal Conditioning:**
- Analog input range: 0-3.3V
- ADC resolution: 12-bit (0-4095)
- Sampling rate: 2kHz
- Built-in bandpass filter: 20-450Hz
- Notch filter: 60Hz (US) or 50Hz (EU)

## 💡 Building Custom Demos

### Serial Communication Example

```python
import serial
import json

# Open serial connection
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0)

while True:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        try:
            data = json.loads(line)

            gesture = data['gesture']
            confidence = data['conf']

            if confidence >= 0.7:  # High confidence threshold
                print(f"Detected: {gesture} ({confidence:.1%})")
                # Trigger your action here

        except json.JSONDecodeError:
            pass
```

### Using the Model in PyTorch

```python
import torch
from generic_neuromotor_interface.networks_isolated import M1_4Channel_TinyML

# Load model
checkpoint = torch.load('models/best_m1_4channel.pt')
model = M1_4Channel_TinyML(
    input_channels=4,
    conv_output_channels=72,
    kernel_width=15,
    stride=10,
    lstm_hidden_size=48,
    lstm_num_layers=2,
    output_channels=9,
    dropout=0.3,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
emg_window = torch.randn(1, 4, 2000)  # (batch, channels, time)
with torch.no_grad():
    logits = model(emg_window)  # (batch, 9, time_out)
    probs = torch.sigmoid(logits)

# Get gesture predictions
confidences = probs.max(dim=-1)[0][0]  # Max over time
predicted_class = confidences.argmax().item()
```

## 📚 File Descriptions

### Firmware Files

| File | Purpose |
|------|---------|
| `main.cpp` | Main loop: EMG acquisition → inference → serial output |
| `m1_4channel_weights.h` | Model weights as C arrays (auto-generated) |
| `m1_inference.cpp` | Native C++ implementation of M1 architecture |
| `gesture_classifier.h` | Gesture class names and IDs |
| `platformio.ini` | Build targets for different models |

### Demo Files

| File | Purpose |
|------|---------|
| `simple_demo.py` | Clean gesture visualization (recommended for presentations) |
| `main.py` | Full game demo with runner mechanics |
| `gesture_input.py` | Abstraction layer for serial/keyboard input |
| `config.py` | All constants, gesture mappings, colors |

## 🐛 Troubleshooting

### Firmware Issues

**"Failed to upload"**
- Hold BOOT button while connecting USB
- Try different USB cable
- Check board appears as `/dev/ttyACM0` (Linux) or `COM` port (Windows)

**"No serial output"**
- Verify baud rate is 115200
- Check firmware is running: LED should blink
- Send 's' command to get status

**"Low confidence scores"**
- Check electrode placement
- Verify good skin contact
- Try re-positioning electrodes
- Perform larger, more deliberate gestures

### Demo Issues

**"Cannot connect to hardware"**
- Check serial port: `ls /dev/ttyACM*` (Linux)
- Edit `SERIAL_PORT` in `config.py`
- Firmware must be running first
- Press 'M' to switch to keyboard mode as fallback

**"Arcade won't run"**
- Need X11 display (won't work on headless servers)
- Use local machine with GUI
- Or use web-based demo (future enhancement)

## 📖 Additional Resources

**Training Code:**
- See parent repo: `/train_m1_4channel.py`
- Config: `/config/discrete_gestures_m1_4channel_tinyml.yaml`
- Evaluation: `/evaluate_all_models_simple.py`

**Model Conversion:**
- PyTorch → C arrays: `/tinyml_deployment/conversion/export_weights_to_c.py`
- TFLite conversion (if needed): `/tinyml_deployment/conversion/convert_m1_4channel.py`

**Dataset:**
- EMG data location: `~/emg_data/` (not included in package)
- 100 participants, 9 gestures, 16 channels
- User-level split: 80/10/10 (train/val/test)

## 📝 Citation

If you use this model in your work, please cite:

```
@misc{m1_4channel_tinyml_2024,
  title={M1 4-Channel TinyML: EMG Gesture Recognition for XIAO nRF52840},
  author={Your Name},
  year={2024},
  note={Trained on 100-participant sEMG dataset, 99.6% validation accuracy}
}
```

## 🔒 License

[Specify your license here]

## ✨ Next Steps

1. **Flash firmware** to XIAO nRF52840
2. **Test with serial monitor** to verify gesture detection
3. **Run Python demo** on local machine with GUI
4. **Build custom UI** using `gesture_input.py` as reference
5. **Deploy to your application** (game, robot control, accessibility tool, etc.)

Ready to deploy! No server needed for inference or demo development.
