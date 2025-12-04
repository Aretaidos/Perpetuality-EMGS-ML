# TinyML Deployment for Seeed Studio XIAO nRF52840 (non-Sense)

Deploy sEMG gesture recognition models to the XIAO nRF52840 microcontroller using TensorFlow Lite Micro.

## Critical Hardware Constraints

| Resource | Available | Notes |
|----------|-----------|-------|
| **RAM** | 256 KB | Only ~220KB usable after stack/heap |
| **Flash** | 1 MB + 2 MB | 1MB internal, 2MB onboard QSPI |
| **CPU** | 64 MHz | ARM Cortex-M4F with FPU |
| **ADC** | 12-bit, 6ch | 200kHz max sample rate |

### Model Options

| Model | Type | Params | RAM Needed | Fits? | Accuracy |
|-------|------|--------|------------|-------|----------|
| **Compressed M1** | CNN+LSTM | ~45K | ~80 KB | **YES** | ~94-96% |
| M2 | CNN-only | ~160K | ~100 KB | **YES** | ~92-94% |
| Original M1 | CNN+LSTM | ~411K | ~398 KB | **NO** | 99.55% |

### Recommended: Compressed M1

The **Compressed M1** model is the recommended choice because:
- Retains LSTM architecture for temporal modeling
- Fits comfortably in 220KB usable RAM
- Expected 94-96% accuracy (down from 99.55% original)

**Architecture comparison:**
```
Original M1 (DOES NOT FIT):
  Conv(7→128) → LSTM(128, 3 layers) → FC(9)
  Parameters: 411,529 | RAM: ~398KB

Compressed M1 (FITS):
  Conv(7→64) → LSTM(32, 1 layer) → FC(9)
  Parameters: ~45,000 | RAM: ~80KB
```

**Key compressions:**
- `conv_output_channels`: 128 → 64
- `lstm_hidden_size`: 128 → 32
- `lstm_num_layers`: 3 → 1
- `window_samples`: 10000 → 1000 (500ms)

## Project Structure

```
tinyml_deployment/
├── README.md                          # This file
├── conversion/
│   ├── convert_model_xiao.py          # Model conversion script
│   ├── convert_m1_to_tflite.py        # Legacy M1 converter
│   └── requirements.txt               # Python dependencies
├── converted_models/                   # Output directory
│   ├── m2_gesture_model.onnx
│   ├── m2_gesture_model.tflite
│   └── m2_model.h                     # C header for embedding
└── platformio_project/
    ├── platformio.ini                 # Build configuration
    ├── src/
    │   ├── main.cpp                   # Main application
    │   ├── gesture_classifier.cpp     # TFLite Micro wrapper
    │   ├── signal_processing.cpp      # EMG filtering
    │   └── emg_acquisition.cpp        # Data acquisition
    └── include/
        ├── gesture_classifier.h
        ├── signal_processing.h
        ├── emg_acquisition.h
        └── m2_model.h                 # Generated model header
```

## Quick Start

### 1. Install Dependencies

```bash
# Python (for model conversion)
cd tinyml_deployment/conversion
pip install -r requirements.txt

# PlatformIO CLI
pip install platformio
```

### 2. Train Compressed M1 Model

```bash
# Train the compressed M1 model
cd generic-neuromotor-interface
python -m generic_neuromotor_interface.train \
    --config config/discrete_gestures_m1_compressed.yaml \
    --output checkpoints_compressed/

# Training will produce checkpoints like:
# checkpoints_compressed/m1_compressed-epoch=50-val_acc=0.950.ckpt
```

### 3. Convert Compressed M1 Model

```bash
cd tinyml_deployment/conversion

# Convert Compressed M1 (RECOMMENDED)
python convert_m1_compressed.py \
    --checkpoint ../../checkpoints_compressed/m1_compressed-epoch=XX-val_acc=X.XXX.ckpt \
    --output-dir ../converted_models \
    --input-samples 1000

# Copy generated header to project
cp ../converted_models/m1_compressed_model.h ../platformio_project/include/
```

**Alternative: Use M2 CNN-only**
```bash
# If M1 doesn't meet needs, convert M2 CNN
python convert_model_xiao.py --model m2 \
    --checkpoint ../../models_gpu_multi_20251124_121916/m2_best.pt \
    --output-dir ../converted_models

cp ../converted_models/m2_model.h ../platformio_project/include/
```

### 4. Build Firmware

```bash
cd tinyml_deployment/platformio_project

# Build for XIAO nRF52840
pio run -e xiao_nrf52840

# Or build CNN-optimized variant
pio run -e xiao_nrf52840_cnn
```

### 5. Flash to Device

```bash
# Connect XIAO nRF52840 via USB-C
# Double-tap reset button to enter bootloader if needed

pio run -e xiao_nrf52840 --target upload
```

### 6. Monitor Output

```bash
pio device monitor

# Or specify port
pio device monitor -p /dev/ttyACM0
```

## PlatformIO Configuration

The correct configuration for XIAO nRF52840 (non-Sense):

```ini
[env:xiao_nrf52840]
platform = https://github.com/Seeed-Studio/platform-seeedboards.git
board = seeed-xiao-afruitnrf52-nrf52840
framework = arduino
lib_archive = no

build_flags =
    -mcpu=cortex-m4
    -mfpu=fpv4-sp-d16
    -mfloat-abi=hard
    -mthumb
    -DARM_MATH_CM4
    -D__FPU_PRESENT=1
    -DTF_LITE_STATIC_MEMORY
    -DTFLITE_MICRO_ENABLED
    -Os
    -ffunction-sections
    -fdata-sections
    -fno-exceptions
    -fno-rtti

lib_deps =
    https://github.com/tensorflow/tflite-micro-arduino-examples.git
```

**Important Notes:**
- Use Seeed's custom platform (not standard `nordicnrf52`)
- Board name is `seeed-xiao-afruitnrf52-nrf52840` (not `xiaoblenrf52840`)
- Must set `lib_archive = no` for TFLite Micro

## Memory Budget

| Component | RAM | Flash |
|-----------|-----|-------|
| Tensor Arena | 100 KB | - |
| Input Buffer (7ch × 1000) | 28 KB | - |
| TFLite Runtime | 30 KB | - |
| Stack/Heap | 50 KB | - |
| **Total Runtime** | **~208 KB** | - |
| Model (INT8) | - | 100-200 KB |
| Application Code | - | ~100 KB |
| **Total** | **208 KB** | **~300 KB** |

Leaves ~48KB RAM and ~700KB flash headroom.

## Serial Commands

When connected via serial monitor (115200 baud):

| Command | Description |
|---------|-------------|
| `s` | Show status (inference count, latency) |
| `r` | Reset statistics |
| `t` | Run test inference with synthetic data |
| `m` | Show memory information |
| `h` | Help |

## Output Format

Gesture detections are output as JSON:

```json
{"n":42,"ms":85,"class":4,"gesture":"thumb_click","conf":0.923,"scores":[0.02,0.01,0.01,0.02,0.92,0.01,0.00,0.01,0.00]}
```

Fields:
- `n`: Inference number
- `ms`: Inference latency (milliseconds)
- `class`: Predicted class (0-8)
- `gesture`: Gesture name
- `conf`: Confidence score (0-1)
- `scores`: All class scores

## Gesture Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | index_press | Index finger press |
| 1 | index_release | Index finger release |
| 2 | middle_press | Middle finger press |
| 3 | middle_release | Middle finger release |
| 4 | thumb_click | Thumb click |
| 5 | thumb_down | Thumb down motion |
| 6 | thumb_in | Thumb in motion |
| 7 | thumb_out | Thumb out motion |
| 8 | thumb_up | Thumb up motion |

## EMG Hardware Setup

### Option 1: Simulated Data (Default)
No hardware required - uses synthetic EMG-like signals for testing.

### Option 2: Analog Input
Connect EMG amplifier outputs to analog pins A0-A5:

```
EMG Frontend    XIAO nRF52840
-----------     -------------
CH1 OUT    -->  A0
CH2 OUT    -->  A1
CH3 OUT    -->  A2
CH4 OUT    -->  A3
CH5 OUT    -->  A4
CH6 OUT    -->  A5
GND        -->  GND
```

Note: Only 6 analog channels available. For 7+ channels, use external ADC.

### Option 3: Serial Input
Send EMG data via serial in CSV format:
```
ch0,ch1,ch2,ch3,ch4,ch5,ch6\n
```

### Option 4: External SPI ADC (Research-grade)

For high-quality EMG, use ADS1299 (8-channel, 24-bit):

```
ADS1299      XIAO nRF52840
-------      -------------
SCLK    -->  D8 (SCK)
MOSI    -->  D10 (MOSI)
MISO    -->  D9 (MISO)
CS      -->  D7
DRDY    -->  D6
GND     -->  GND
VCC     -->  3.3V
```

## Troubleshooting

### Build Errors

**"Board not found"**
```
Error: Unknown board ID 'xiaoblenrf52840'
```
Solution: Use Seeed's platform and correct board name:
```ini
platform = https://github.com/Seeed-Studio/platform-seeedboards.git
board = seeed-xiao-afruitnrf52-nrf52840
```

**"No model header found"**
```
warning: No model header found! Using placeholder.
```
Solution: Convert model and copy header:
```bash
python convert_model_xiao.py --model m2
cp converted_models/m2_model.h platformio_project/include/
```

**"AllocateTensors() failed"**
```
ERROR: AllocateTensors() failed
Arena size: 102400 bytes
```
Solution: Increase `TENSOR_ARENA_SIZE` in `gesture_classifier.h` or use smaller model.

### Runtime Errors

**Classifier not ready (fast LED blink)**
- Model not loaded correctly
- Check serial output for error messages
- Verify model header is included

**No serial output**
- Check baud rate (115200)
- Try different USB cable (data, not charge-only)
- Double-tap reset to enter bootloader

### Memory Issues

**Inference too slow (>200ms)**
- Enable FPU flags in build
- Use INT8 quantized model
- Reduce input window size

**Out of memory**
- Use M2 CNN instead of M1 LSTM
- Reduce `TENSOR_ARENA_SIZE`
- Reduce `WINDOW_SAMPLES`

## Performance Expectations

| Metric | M2 CNN | Notes |
|--------|--------|-------|
| Inference Latency | 50-100 ms | At 64MHz with INT8 |
| Throughput | ~10-20 Hz | Continuous inference |
| Power (Active) | ~30-50 mW | During inference |
| Power (Idle) | ~5 µA | Deep sleep |
| Model Size | 100-200 KB | INT8 quantized |

## References

- [Seeed XIAO nRF52840 Wiki](https://wiki.seeedstudio.com/XIAO_BLE/)
- [XIAO with PlatformIO](https://wiki.seeedstudio.com/xiao_nrf52840_with_platform_io/)
- [TFLite Micro Arduino Examples](https://github.com/tensorflow/tflite-micro-arduino-examples)
- [nRF52840 Product Spec](https://infocenter.nordicsemi.com/topic/ps_nrf52840/keyfeatures_html5.html)
- [TFLite for Microcontrollers](https://ai.google.dev/edge/litert/microcontrollers)

## License

Part of the Generic Neuromotor Interface project.
