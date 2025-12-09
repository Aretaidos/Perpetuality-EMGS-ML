"""
Signal Drift - Gesture Input Handler
Handles both serial hardware input and keyboard simulation
"""

import json
import time
from typing import Dict, Optional, Set
import serial

from config import (
    SERIAL_PORT,
    SERIAL_BAUDRATE,
    CONFIDENCE_THRESHOLD_DETECT,
    GESTURE_NAMES,
)


class GestureInput:
    """Handles gesture input from hardware (serial) or keyboard simulation"""

    def __init__(self, mode='keyboard'):
        """
        Args:
            mode: 'hardware' or 'keyboard'
        """
        self.mode = mode
        self.serial = None
        self.last_gesture_data = None
        self.emg_amplitude = 0.0  # Simulated EMG signal strength (0-1)

        if mode == 'hardware':
            try:
                self.serial = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0)
                print(f"✓ Connected to hardware on {SERIAL_PORT}")
            except Exception as e:
                print(f"✗ Failed to connect to hardware: {e}")
                print("  Falling back to keyboard mode")
                self.mode = 'keyboard'

    def toggle_mode(self):
        """Switch between hardware and keyboard modes"""
        if self.mode == 'keyboard':
            # Try to connect to hardware
            try:
                self.serial = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0)
                self.mode = 'hardware'
                print(f"✓ Switched to HARDWARE mode")
                return True
            except Exception as e:
                print(f"✗ Cannot switch to hardware: {e}")
                return False
        else:
            # Switch to keyboard
            if self.serial:
                self.serial.close()
                self.serial = None
            self.mode = 'keyboard'
            print(f"✓ Switched to KEYBOARD mode")
            return True

    def poll_hardware(self) -> Optional[Dict]:
        """
        Poll serial port for gesture detection JSON

        Returns:
            Dict with gesture data or None
            Example: {"class": 4, "gesture": "thumb_click", "conf": 0.92, "scores": [...]}
        """
        if not self.serial or not self.serial.in_waiting:
            return None

        try:
            line = self.serial.readline().decode('utf-8').strip()
            if not line:
                return None

            data = json.loads(line)

            # Validate data structure
            if 'gesture' in data and 'conf' in data and 'scores' in data:
                self.last_gesture_data = data
                return data

        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
            # Ignore malformed data
            pass
        except Exception as e:
            print(f"Serial error: {e}")
            # Try to reconnect
            self.serial.close()
            time.sleep(0.1)
            try:
                self.serial = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0)
            except:
                self.mode = 'keyboard'
                print("Lost hardware connection, switched to keyboard mode")

        return None

    def get_gesture_confidences(self) -> Dict[str, float]:
        """
        Get current confidence scores for all 9 gestures

        Returns:
            Dict mapping gesture name -> confidence (0-1)
        """
        if self.mode == 'hardware' and self.last_gesture_data:
            # Use actual scores from hardware
            scores = self.last_gesture_data.get('scores', [0.0] * 9)
            return {
                gesture: scores[i] if i < len(scores) else 0.0
                for i, gesture in enumerate(GESTURE_NAMES)
            }
        else:
            # Keyboard mode: all zeros except recently triggered
            return {gesture: 0.0 for gesture in GESTURE_NAMES}

    def get_detected_gesture(self) -> Optional[Dict]:
        """
        Get currently detected gesture (if confidence > threshold)

        Returns:
            Dict with 'gesture' name and 'confidence', or None
        """
        if self.mode == 'hardware' and self.last_gesture_data:
            conf = self.last_gesture_data.get('conf', 0.0)
            gesture = self.last_gesture_data.get('gesture', '')

            if conf >= CONFIDENCE_THRESHOLD_DETECT and gesture in GESTURE_NAMES:
                return {'gesture': gesture, 'confidence': conf}

        return None

    def get_emg_amplitude(self) -> float:
        """
        Get current EMG signal amplitude (0-1)
        For hardware: derived from confidence scores
        For keyboard: manually controlled

        Returns:
            Float 0-1 representing signal strength
        """
        if self.mode == 'hardware' and self.last_gesture_data:
            # Use max of all confidence scores as proxy for EMG amplitude
            scores = self.last_gesture_data.get('scores', [0.0])
            return max(scores)
        else:
            # Keyboard mode: use simulated amplitude
            return self.emg_amplitude

    def set_emg_amplitude(self, value: float):
        """Set simulated EMG amplitude (keyboard mode only)"""
        self.emg_amplitude = max(0.0, min(1.0, value))

    def update(self):
        """Poll for new data (call every frame)"""
        if self.mode == 'hardware':
            self.poll_hardware()

    def close(self):
        """Clean up serial connection"""
        if self.serial:
            self.serial.close()


class KeyboardGestureSimulator:
    """Simulates gesture detections via keyboard for testing"""

    def __init__(self):
        self.triggered_gestures = {}  # gesture -> (timestamp, confidence)
        self.trigger_duration = 0.5  # How long a keyboard trigger lasts

    def trigger_gesture(self, gesture_name: str, confidence: float = 0.95):
        """Simulate detecting a gesture"""
        if gesture_name in GESTURE_NAMES:
            self.triggered_gestures[gesture_name] = (time.time(), confidence)

    def get_active_gesture(self) -> Optional[Dict]:
        """Get currently active simulated gesture"""
        current_time = time.time()

        # Find most recently triggered gesture that's still active
        for gesture, (timestamp, conf) in self.triggered_gestures.items():
            if current_time - timestamp < self.trigger_duration:
                return {'gesture': gesture, 'confidence': conf}

        return None

    def get_confidences(self) -> Dict[str, float]:
        """Get confidence scores showing recent triggers"""
        current_time = time.time()
        confidences = {}

        for gesture in GESTURE_NAMES:
            if gesture in self.triggered_gestures:
                timestamp, conf = self.triggered_gestures[gesture]
                # Decay confidence over time
                age = current_time - timestamp
                if age < self.trigger_duration:
                    decay_factor = 1.0 - (age / self.trigger_duration)
                    confidences[gesture] = conf * decay_factor
                else:
                    confidences[gesture] = 0.0
            else:
                confidences[gesture] = 0.0

        return confidences
