#!/usr/bin/env python3
"""
EMG Gesture Demo - Simple & Visual

Shows:
1. Hand diagram with muscle activation visualization
2. Live EMG traces (4 channels)
3. Gesture confidence bars
4. Interactive objects that respond to gestures

When gesture detected → Hand lights up + object reacts

Controls:
    Keyboard: 1-9 to trigger gestures, UP to simulate EMG
    Hardware: Automatic from XIAO nRF52840
    M: Toggle mode, ESC: Quit
"""

import arcade
import math
import time
from typing import Dict, Optional

from config import *
from gesture_input import GestureInput, KeyboardGestureSimulator


class GestureDemo(arcade.Window):
    """Simple gesture visualization demo"""

    def __init__(self, start_mode='keyboard'):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "EMG Gesture Demo")
        arcade.set_background_color((15, 15, 20))

        # Input
        self.gesture_input = GestureInput(mode=start_mode)
        self.keyboard_sim = KeyboardGestureSimulator()

        # Visual state
        self.current_gesture = None
        self.gesture_start_time = 0
        self.gesture_flash_duration = 1.0

        # Animated objects (one per gesture)
        self.objects = self.create_gesture_objects()

    def create_gesture_objects(self):
        """Create visual objects for each gesture"""
        objects = {}

        # Position objects in a circle around center
        center_x = SCREEN_WIDTH * 0.7
        center_y = SCREEN_HEIGHT * 0.5
        radius = 200

        for i, gesture in enumerate(GESTURE_NAMES):
            angle = (i / 9) * 2 * math.pi - math.pi/2  # Start at top
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            objects[gesture] = {
                'x': x,
                'y': y,
                'base_size': 30,
                'current_size': 30,
                'color': POWERUP_CONFIGS[gesture]['color'],
                'active': False,
                'activation_time': 0,
            }

        return objects

    def on_update(self, delta_time: float):
        """Update"""
        self.gesture_input.update()

        # Get detected gesture
        if self.gesture_input.mode == 'hardware':
            detected = self.gesture_input.get_detected_gesture()
        else:
            detected = self.keyboard_sim.get_active_gesture()

        # Update current gesture
        if detected and detected['confidence'] >= CONFIDENCE_THRESHOLD_POWERUP:
            if detected['gesture'] != self.current_gesture:
                self.current_gesture = detected['gesture']
                self.gesture_start_time = time.time()

                # Activate corresponding object
                obj = self.objects[self.current_gesture]
                obj['active'] = True
                obj['activation_time'] = time.time()

        # Update objects (pulse animation)
        current_time = time.time()
        for gesture, obj in self.objects.items():
            if obj['active']:
                # Pulse for 1 second
                age = current_time - obj['activation_time']
                if age < 1.0:
                    pulse = 1.0 + 0.5 * math.sin(age * 10)  # Fast pulse
                    obj['current_size'] = obj['base_size'] * pulse
                else:
                    obj['active'] = False
                    obj['current_size'] = obj['base_size']

    def on_draw(self):
        """Render"""
        self.start_render()

        # Split screen: Left = hand visualization, Right = gesture objects
        self.draw_hand_visualization()
        self.draw_gesture_objects()
        self.draw_emg_traces()
        self.draw_confidence_bars()
        self.draw_ui()

    def draw_hand_visualization(self):
        """Draw hand diagram with muscle activation"""
        # Title
        arcade.draw_text(
            "MUSCLE ACTIVATION",
            20, SCREEN_HEIGHT - 30,
            (200, 200, 255), 16, bold=True
        )

        # Get EMG amplitude for each channel
        confidences = self.gesture_input.get_gesture_confidences() if self.gesture_input.mode == 'hardware' else self.keyboard_sim.get_confidences()

        # Simple hand representation
        hand_x = 200
        hand_y = SCREEN_HEIGHT - 250

        # Draw forearm
        arcade.draw_rectangle_filled(
            hand_x, hand_y - 100,
            60, 120,
            (80, 80, 100)
        )

        # Draw palm
        arcade.draw_circle_filled(hand_x, hand_y, 50, (100, 100, 120))

        # Draw fingers (simplified)
        fingers = [
            ('index', -30, 80),
            ('middle', 0, 90),
            ('ring', 30, 80),
            ('thumb', -60, 20),
        ]

        for name, offset_x, length in fingers:
            # Check if this finger is active
            active_press = any(
                conf > 0.5 and name in gesture
                for gesture, conf in confidences.items()
            )

            color = (255, 150, 50) if active_press else (120, 120, 140)
            width = 20 if active_press else 15

            arcade.draw_line(
                hand_x + offset_x, hand_y,
                hand_x + offset_x, hand_y + length,
                color, width
            )

        # Draw muscle labels
        muscle_y = hand_y - 200
        arcade.draw_text("Ch7: Index Flexor", 20, muscle_y, (255, 255, 255), 10)
        arcade.draw_text("Ch8: Ring Flexor", 20, muscle_y - 20, (255, 255, 255), 10)
        arcade.draw_text("Ch13: Index Extensor", 20, muscle_y - 40, (255, 255, 255), 10)
        arcade.draw_text("Ch15: Ring Extensor", 20, muscle_y - 60, (255, 255, 255), 10)

        # Current gesture overlay
        if self.current_gesture:
            age = time.time() - self.gesture_start_time
            if age < self.gesture_flash_duration:
                alpha = int(255 * (1.0 - age / self.gesture_flash_duration))
                gesture_name = GESTURE_DISPLAY_NAMES.get(self.current_gesture, self.current_gesture)

                arcade.draw_text(
                    gesture_name.upper(),
                    hand_x, hand_y - 150,
                    (*POWERUP_CONFIGS[self.current_gesture]['color'], alpha),
                    20, anchor_x="center", bold=True
                )

    def draw_gesture_objects(self):
        """Draw interactive gesture objects"""
        # Title
        arcade.draw_text(
            "GESTURE ACTIONS",
            SCREEN_WIDTH * 0.55, SCREEN_HEIGHT - 30,
            (200, 200, 255), 16, bold=True
        )

        # Draw objects
        for gesture, obj in self.objects.items():
            # Circle
            arcade.draw_circle_filled(
                obj['x'], obj['y'],
                obj['current_size'],
                obj['color']
            )

            # Gesture label
            label = GESTURE_DISPLAY_NAMES.get(gesture, gesture)
            arcade.draw_text(
                label,
                obj['x'], obj['y'] - 50,
                (200, 200, 200), 10,
                anchor_x="center"
            )

            # Active indicator
            if obj['active']:
                arcade.draw_circle_outline(
                    obj['x'], obj['y'],
                    obj['current_size'] + 10,
                    (255, 255, 255), 3
                )

    def draw_emg_traces(self):
        """Draw 4-channel EMG traces"""
        trace_x = 20
        trace_y = 300
        trace_width = 300
        trace_height = 40

        arcade.draw_text(
            "LIVE EMG (4 Channels)",
            trace_x, trace_y + 200,
            (200, 200, 255), 12, bold=True
        )

        # Placeholder: would need real EMG samples for actual traces
        # For now, show bars representing amplitude
        channel_names = ["Ch7 (Index Flex)", "Ch8 (Ring Flex)",
                         "Ch13 (Index Ext)", "Ch15 (Ring Ext)"]

        emg_amp = self.gesture_input.get_emg_amplitude()

        for i, name in enumerate(channel_names):
            y = trace_y + (3 - i) * (trace_height + 10)

            # Background
            arcade.draw_rectangle_filled(
                trace_x + trace_width/2, y,
                trace_width, trace_height,
                (40, 40, 50)
            )

            # Simulated activity
            activity = emg_amp * (0.7 + 0.3 * math.sin(time.time() * 3 + i))
            bar_width = trace_width * activity

            arcade.draw_rectangle_filled(
                trace_x + bar_width/2, y,
                bar_width, trace_height,
                (100, 200, 100)
            )

            # Label
            arcade.draw_text(
                name,
                trace_x + trace_width + 10, y - 5,
                (180, 180, 180), 9
            )

    def draw_confidence_bars(self):
        """Draw gesture confidence bars"""
        bar_x = SCREEN_WIDTH - 320
        bar_y = SCREEN_HEIGHT - 80
        bar_width = 200

        arcade.draw_text(
            "GESTURE CONFIDENCE",
            bar_x, bar_y + 30,
            (200, 200, 255), 12, bold=True
        )

        # Get confidences
        if self.gesture_input.mode == 'hardware':
            confidences = self.gesture_input.get_gesture_confidences()
        else:
            confidences = self.keyboard_sim.get_confidences()

        for i, gesture in enumerate(GESTURE_NAMES):
            conf = confidences.get(gesture, 0.0)
            y = bar_y - i * 20

            # Label
            label = GESTURE_DISPLAY_NAMES.get(gesture, gesture)
            arcade.draw_text(
                f"{label:10}",
                bar_x, y,
                (180, 180, 180), 9
            )

            # Bar background
            arcade.draw_rectangle_filled(
                bar_x + 100 + bar_width/2, y + 5,
                bar_width, 12,
                (50, 50, 50)
            )

            # Bar fill
            if conf > 0:
                fill_width = bar_width * conf
                color = (50, 255, 50) if conf >= 0.7 else (100, 150, 100)

                arcade.draw_rectangle_filled(
                    bar_x + 100 + fill_width/2, y + 5,
                    fill_width, 12,
                    color
                )

            # Percentage
            arcade.draw_text(
                f"{conf*100:3.0f}%",
                bar_x + 100 + bar_width + 10, y,
                (200, 200, 200), 9
            )

    def draw_ui(self):
        """Draw mode indicator and help"""
        # Mode
        mode_text = f"Mode: {'HARDWARE' if self.gesture_input.mode == 'hardware' else 'KEYBOARD'}"
        mode_color = (50, 255, 50) if self.gesture_input.mode == 'hardware' else (100, 150, 255)
        arcade.draw_text(
            mode_text,
            20, 20,
            mode_color, 12, bold=True
        )

        # Help
        if self.gesture_input.mode == 'keyboard':
            arcade.draw_text(
                "Keys: UP=EMG signal, 1-9=gestures, M=toggle mode",
                20, 5,
                (150, 150, 150), 9
            )

    def on_key_press(self, key, modifiers):
        """Handle keys"""
        if key == arcade.key.ESCAPE:
            self.close()
        elif key == arcade.key.M:
            self.gesture_input.toggle_mode()

        # Keyboard simulation
        if self.gesture_input.mode == 'keyboard':
            if key == arcade.key.UP:
                current = self.gesture_input.get_emg_amplitude()
                self.gesture_input.set_emg_amplitude(min(1.0, current + 0.3))
            elif key == arcade.key.DOWN:
                current = self.gesture_input.get_emg_amplitude()
                self.gesture_input.set_emg_amplitude(max(0.0, current - 0.3))
            elif key in KEYBOARD_MAP:
                action = KEYBOARD_MAP[key]
                if action in GESTURE_NAMES:
                    self.keyboard_sim.trigger_gesture(action, confidence=0.95)

    def on_close(self):
        """Cleanup"""
        self.gesture_input.close()


def main():
    print("=" * 60)
    print("EMG GESTURE DEMO - Simple & Visual")
    print("=" * 60)
    print("\nShows:")
    print("  • Hand diagram with muscle activation")
    print("  • Live EMG traces (4 channels)")
    print("  • Gesture confidence bars")
    print("  • Interactive objects responding to gestures")
    print("\nControls:")
    print("  M:        Toggle hardware/keyboard mode")
    print("  UP/DOWN:  Simulate EMG (keyboard mode)")
    print("  1-9:      Trigger gestures (keyboard mode)")
    print("  ESC:      Quit")
    print("=" * 60)

    demo = GestureDemo(start_mode='keyboard')
    arcade.run()


if __name__ == "__main__":
    main()
