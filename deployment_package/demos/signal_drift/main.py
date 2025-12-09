#!/usr/bin/env python3
"""
Signal Drift - EMG Gesture Runner Demo

A side-scrolling runner where EMG signal strength controls vertical position.
Confident gesture detections trigger power-ups. Shows live ML confidence scores.

Controls:
    Keyboard Mode:
        UP/DOWN: Simulate EMG signal strength
        1-9: Trigger gesture detections
        M: Toggle hardware/keyboard mode
        P: Pause
        ESC: Quit

    Hardware Mode:
        Any muscle activation: Orb rises
        Relaxed state: Orb falls
        Confident gestures (>70%): Trigger power-ups

Usage:
    python main.py                    # Start in keyboard mode
    python main.py --hardware         # Start in hardware mode
"""

import argparse
import arcade
import random
import time
from typing import List, Optional, Dict

from config import *
from gesture_input import GestureInput, KeyboardGestureSimulator


class PowerUp:
    """Active power-up with timer"""

    def __init__(self, gesture_name: str, config: Dict):
        self.gesture_name = gesture_name
        self.name = config['name']
        self.color = config['color']
        self.duration = config['duration']
        self.effect = config['effect']
        self.multiplier = config['multiplier']
        self.time_remaining = config['duration']

    def update(self, delta_time: float) -> bool:
        """Update power-up, return True if still active"""
        self.time_remaining -= delta_time
        return self.time_remaining > 0


class Obstacle:
    """Obstacle in the tunnel"""

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self):
        arcade.draw_rectangle_filled(
            self.x, self.y, self.width, self.height, COLOR_OBSTACLE
        )

    def update(self, speed: float):
        self.x -= speed


class Coin:
    """Collectible coin"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.radius = 8

    def draw(self):
        arcade.draw_circle_filled(self.x, self.y, self.radius, COLOR_COIN)

    def update(self, speed: float):
        self.x -= speed


class SignalDriftGame(arcade.Window):
    """Main game window"""

    def __init__(self, start_mode='keyboard'):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(COLOR_BACKGROUND)

        # Input handling
        self.gesture_input = GestureInput(mode=start_mode)
        self.keyboard_sim = KeyboardGestureSimulator()

        # Game state
        self.paused = False
        self.game_over = False
        self.score = 0
        self.distance = 0
        self.high_score = 0

        # Orb (player)
        self.orb_x = 200
        self.orb_y = GAME_AREA_HEIGHT // 2
        self.orb_velocity_y = 0
        self.orb_radius = ORB_RADIUS

        # Power-ups
        self.active_powerups: List[PowerUp] = []
        self.gesture_streak = 0

        # Obstacles and coins
        self.obstacles: List[Obstacle] = []
        self.coins: List[Coin] = []
        self.time_since_obstacle = 0
        self.time_since_coin = 0

        # Visual effects
        self.last_detection_time = 0
        self.detection_flash_duration = 0.5

        # Performance
        self.frame_count = 0
        self.fps = 60

    def setup(self):
        """Initialize/reset game"""
        self.game_over = False
        self.score = 0
        self.distance = 0
        self.orb_y = GAME_AREA_HEIGHT // 2
        self.orb_velocity_y = 0
        self.active_powerups.clear()
        self.obstacles.clear()
        self.coins.clear()
        self.gesture_streak = 0
        self.time_since_obstacle = 0
        self.time_since_coin = 0

    def on_update(self, delta_time: float):
        """Update game state"""
        if self.paused or self.game_over:
            return

        self.frame_count += 1

        # Update input
        self.gesture_input.update()

        # Get EMG amplitude
        emg_amplitude = self.gesture_input.get_emg_amplitude()

        # Apply lift force based on EMG
        if emg_amplitude > 0.2:  # Threshold to activate
            self.orb_velocity_y += ORB_LIFT_FORCE * emg_amplitude
        else:
            # Gravity pulls down when relaxed
            self.orb_velocity_y -= ORB_GRAVITY

        # Clamp velocity
        self.orb_velocity_y = max(-ORB_MAX_SPEED, min(ORB_MAX_SPEED, self.orb_velocity_y))

        # Update orb position
        self.orb_y += self.orb_velocity_y

        # Keep orb in bounds
        if self.orb_y < self.orb_radius:
            self.orb_y = self.orb_radius
            self.orb_velocity_y = 0
        elif self.orb_y > GAME_AREA_HEIGHT - self.orb_radius:
            self.orb_y = GAME_AREA_HEIGHT - self.orb_radius
            self.orb_velocity_y = 0

        # Check for gesture detections
        if self.gesture_input.mode == 'hardware':
            detected = self.gesture_input.get_detected_gesture()
        else:
            detected = self.keyboard_sim.get_active_gesture()

        if detected and detected['confidence'] >= CONFIDENCE_THRESHOLD_POWERUP:
            self.trigger_powerup(detected['gesture'], detected['confidence'])

        # Update power-ups
        self.active_powerups = [p for p in self.active_powerups if p.update(delta_time)]

        # Get current scroll speed (modified by power-ups)
        scroll_speed = self.get_scroll_speed()

        # Update distance
        self.distance += scroll_speed * delta_time * 10

        # Spawn obstacles
        self.time_since_obstacle += delta_time
        if self.time_since_obstacle >= OBSTACLE_SPAWN_INTERVAL:
            self.spawn_obstacle()
            self.time_since_obstacle = 0

        # Spawn coins
        self.time_since_coin += delta_time
        if self.time_since_coin >= COIN_SPAWN_INTERVAL:
            self.spawn_coin()
            self.time_since_coin = 0

        # Update obstacles
        for obs in self.obstacles:
            obs.update(scroll_speed)

        # Update coins
        for coin in self.coins:
            coin.update(scroll_speed)

        # Remove off-screen objects
        self.obstacles = [o for o in self.obstacles if o.x > -o.width]
        self.coins = [c for c in self.coins if c.x > -c.radius]

        # Check collisions
        self.check_collisions()

        # Update score
        self.score = int(self.distance) + (self.gesture_streak * 100)

    def get_scroll_speed(self) -> float:
        """Get current scroll speed (modified by power-ups)"""
        speed = TUNNEL_SCROLL_SPEED

        for powerup in self.active_powerups:
            if powerup.effect == 'speed':
                speed *= powerup.multiplier
            elif powerup.effect == 'slow_mo':
                speed *= powerup.multiplier

        return speed

    def trigger_powerup(self, gesture_name: str, confidence: float):
        """Trigger a power-up from confident gesture detection"""
        if gesture_name not in POWERUP_CONFIGS:
            return

        # Check if already active
        for powerup in self.active_powerups:
            if powerup.gesture_name == gesture_name:
                # Reset timer
                powerup.time_remaining = powerup.duration
                return

        # Add new power-up
        config = POWERUP_CONFIGS[gesture_name]
        powerup = PowerUp(gesture_name, config)
        self.active_powerups.append(powerup)

        # Visual feedback
        self.last_detection_time = time.time()
        self.gesture_streak += 1

        print(f"✨ Power-up activated: {powerup.name} ({confidence:.1%} confidence)")

    def spawn_obstacle(self):
        """Spawn a new obstacle"""
        gap_y = random.randint(100, GAME_AREA_HEIGHT - 100)
        gap_height = random.randint(120, 180)

        # Top obstacle
        top_obs = Obstacle(
            x=SCREEN_WIDTH + 50,
            y=GAME_AREA_HEIGHT - (GAME_AREA_HEIGHT - gap_y - gap_height/2) / 2,
            width=40,
            height=GAME_AREA_HEIGHT - gap_y - gap_height/2
        )
        self.obstacles.append(top_obs)

        # Bottom obstacle
        bottom_obs = Obstacle(
            x=SCREEN_WIDTH + 50,
            y=(gap_y - gap_height/2) / 2,
            width=40,
            height=gap_y - gap_height/2
        )
        self.obstacles.append(bottom_obs)

    def spawn_coin(self):
        """Spawn a collectible coin"""
        coin_y = random.randint(50, GAME_AREA_HEIGHT - 50)
        coin = Coin(x=SCREEN_WIDTH + 20, y=coin_y)
        self.coins.append(coin)

    def check_collisions(self):
        """Check for collisions with obstacles and coins"""
        # Check if shield is active
        has_shield = any(p.effect == 'shield' for p in self.active_powerups)

        # Check obstacles
        if not has_shield:
            for obs in self.obstacles:
                if (abs(self.orb_x - obs.x) < obs.width/2 + self.orb_radius and
                    abs(self.orb_y - obs.y) < obs.height/2 + self.orb_radius):
                    self.game_over = True
                    if self.score > self.high_score:
                        self.high_score = self.score
                    return

        # Check coins
        for coin in self.coins[:]:
            if (abs(self.orb_x - coin.x) < coin.radius + self.orb_radius and
                abs(self.orb_y - coin.y) < coin.radius + self.orb_radius):
                self.coins.remove(coin)
                points = 10
                if any(p.effect == 'double_points' for p in self.active_powerups):
                    points *= 2
                self.score += points

    def on_draw(self):
        """Render everything"""
        self.start_render()

        # Draw game area
        self.draw_game_area()

        # Draw confidence display
        self.draw_confidence_area()

        # Draw UI overlays
        self.draw_ui()

    def draw_game_area(self):
        """Draw the main game area"""
        # Draw tunnel walls
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH/2, GAME_AREA_HEIGHT, SCREEN_WIDTH, 4, COLOR_TUNNEL_TOP
        )
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH/2, 0, SCREEN_WIDTH, 4, COLOR_TUNNEL_BOTTOM
        )

        # Draw obstacles
        for obs in self.obstacles:
            obs.draw()

        # Draw coins
        for coin in self.coins:
            coin.draw()

        # Draw orb
        orb_color = COLOR_ORB
        # Modify color if power-ups active
        if self.active_powerups:
            orb_color = self.active_powerups[0].color

        arcade.draw_circle_filled(self.orb_x, self.orb_y, self.orb_radius, orb_color)

        # Draw shield effect
        if any(p.effect == 'shield' for p in self.active_powerups):
            arcade.draw_circle_outline(self.orb_x, self.orb_y, self.orb_radius + 10,
                                        (50, 255, 50, 128), 3)

    def draw_confidence_area(self):
        """Draw gesture confidence waveforms"""
        # Background
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH/2, GAME_AREA_HEIGHT + CONFIDENCE_AREA_HEIGHT/2,
            SCREEN_WIDTH, CONFIDENCE_AREA_HEIGHT, (20, 20, 30)
        )

        # Title
        arcade.draw_text(
            "GESTURE CONFIDENCE (Live ML Output):",
            10, GAME_AREA_HEIGHT + CONFIDENCE_AREA_HEIGHT - 25,
            (200, 200, 200), 12, bold=True
        )

        # Get confidences
        if self.gesture_input.mode == 'hardware':
            confidences = self.gesture_input.get_gesture_confidences()
        else:
            confidences = self.keyboard_sim.get_confidences()

        # Draw each gesture row
        y = GAME_AREA_HEIGHT + CONFIDENCE_AREA_HEIGHT - 50
        row_height = 18

        for i, gesture_name in enumerate(GESTURE_NAMES):
            conf = confidences.get(gesture_name, 0.0)
            display_name = GESTURE_DISPLAY_NAMES.get(gesture_name, gesture_name)

            # Gesture label
            arcade.draw_text(
                f"{display_name:12}",
                10, y - i * row_height,
                (180, 180, 180), 10
            )

            # Confidence bar
            bar_width = 150
            bar_x = 120
            filled_width = bar_width * conf

            # Background bar
            arcade.draw_rectangle_filled(
                bar_x + bar_width/2, y - i * row_height + 5,
                bar_width, 12, (50, 50, 50)
            )

            # Filled bar
            if filled_width > 0:
                color = (50, 200, 50) if conf >= CONFIDENCE_THRESHOLD_POWERUP else (100, 100, 150)
                arcade.draw_rectangle_filled(
                    bar_x + filled_width/2, y - i * row_height + 5,
                    filled_width, 12, color
                )

            # Percentage
            arcade.draw_text(
                f"{conf*100:3.0f}%",
                bar_x + bar_width + 10, y - i * row_height,
                (200, 200, 200), 10
            )

            # Detection badge
            if conf >= CONFIDENCE_THRESHOLD_POWERUP:
                arcade.draw_text(
                    "✓ DETECTED!",
                    bar_x + bar_width + 60, y - i * row_height,
                    (50, 255, 50), 10, bold=True
                )

        # EMG amplitude indicator
        emg = self.gesture_input.get_emg_amplitude()
        arcade.draw_text(
            f"RAW EMG Amplitude: {emg*100:.0f}%",
            10, GAME_AREA_HEIGHT + 10,
            (255, 255, 100), 12, bold=True
        )

    def draw_ui(self):
        """Draw UI overlays (score, mode, power-ups)"""
        # Score
        arcade.draw_text(
            f"Score: {self.score:,}",
            10, SCREEN_HEIGHT - 30,
            (255, 255, 255), 16, bold=True
        )

        # Streak
        if self.gesture_streak > 0:
            arcade.draw_text(
                f"Gesture Streak: {self.gesture_streak}x",
                10, SCREEN_HEIGHT - 55,
                (255, 215, 0), 14, bold=True
            )

        # Mode indicator
        mode_text = f"Mode: {'HARDWARE' if self.gesture_input.mode == 'hardware' else 'KEYBOARD'}"
        mode_color = (50, 255, 50) if self.gesture_input.mode == 'hardware' else (100, 150, 255)
        arcade.draw_text(
            mode_text,
            SCREEN_WIDTH - 200, SCREEN_HEIGHT - 30,
            mode_color, 14, bold=True
        )

        # Active power-ups
        if self.active_powerups:
            y_offset = SCREEN_HEIGHT - 80
            for powerup in self.active_powerups:
                arcade.draw_text(
                    f"{powerup.name}: {powerup.time_remaining:.1f}s",
                    10, y_offset,
                    powerup.color, 12
                )
                y_offset -= 20

        # Game over
        if self.game_over:
            arcade.draw_rectangle_filled(
                SCREEN_WIDTH/2, GAME_AREA_HEIGHT/2,
                400, 200, (0, 0, 0, 200)
            )
            arcade.draw_text(
                "GAME OVER",
                SCREEN_WIDTH/2, GAME_AREA_HEIGHT/2 + 40,
                (255, 50, 50), 32, anchor_x="center", bold=True
            )
            arcade.draw_text(
                f"Final Score: {self.score:,}",
                SCREEN_WIDTH/2, GAME_AREA_HEIGHT/2,
                (255, 255, 255), 20, anchor_x="center"
            )
            arcade.draw_text(
                f"High Score: {self.high_score:,}",
                SCREEN_WIDTH/2, GAME_AREA_HEIGHT/2 - 30,
                (255, 215, 0), 16, anchor_x="center"
            )
            arcade.draw_text(
                "Press SPACE to restart or ESC to quit",
                SCREEN_WIDTH/2, GAME_AREA_HEIGHT/2 - 60,
                (200, 200, 200), 12, anchor_x="center"
            )

        # Pause
        if self.paused:
            arcade.draw_text(
                "PAUSED",
                SCREEN_WIDTH/2, GAME_AREA_HEIGHT/2,
                (255, 255, 255), 32, anchor_x="center", bold=True
            )

    def on_key_press(self, key, modifiers):
        """Handle key presses"""
        if key == arcade.key.ESCAPE:
            self.close()

        elif key == arcade.key.P:
            self.paused = not self.paused

        elif key == arcade.key.M:
            # Toggle mode
            success = self.gesture_input.toggle_mode()

        elif key == arcade.key.SPACE and self.game_over:
            self.setup()

        # Keyboard simulation controls
        if self.gesture_input.mode == 'keyboard':
            if key == arcade.key.UP:
                # Increase EMG signal
                current = self.gesture_input.get_emg_amplitude()
                self.gesture_input.set_emg_amplitude(min(1.0, current + 0.2))

            elif key == arcade.key.DOWN:
                # Decrease EMG signal
                current = self.gesture_input.get_emg_amplitude()
                self.gesture_input.set_emg_amplitude(max(0.0, current - 0.2))

            # Gesture triggers (1-9 keys)
            elif key in KEYBOARD_MAP:
                action = KEYBOARD_MAP[key]
                if action in GESTURE_NAMES:
                    self.keyboard_sim.trigger_gesture(action)

    def on_close(self):
        """Clean up on exit"""
        self.gesture_input.close()
        super().on_close()


def main():
    parser = argparse.ArgumentParser(description='Signal Drift - EMG Gesture Runner')
    parser.add_argument('--hardware', action='store_true',
                        help='Start in hardware mode (default: keyboard)')
    args = parser.parse_args()

    start_mode = 'hardware' if args.hardware else 'keyboard'

    print("=" * 60)
    print("SIGNAL DRIFT - EMG Gesture Runner Demo")
    print("=" * 60)
    print(f"Starting in {start_mode.upper()} mode")
    print()
    print("Controls:")
    print("  M:        Toggle hardware/keyboard mode")
    print("  P:        Pause")
    print("  ESC:      Quit")
    if start_mode == 'keyboard':
        print()
        print("Keyboard Mode:")
        print("  UP/DOWN:  Simulate EMG signal strength")
        print("  1-9:      Trigger gesture detections")
    print("=" * 60)

    game = SignalDriftGame(start_mode=start_mode)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
