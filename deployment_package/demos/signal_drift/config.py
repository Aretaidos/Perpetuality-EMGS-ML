"""
Signal Drift - Configuration
Constants, gesture mappings, and visual parameters
"""

import arcade

# Screen dimensions
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Signal Drift - EMG Gesture Runner"

# Game area split
GAME_AREA_HEIGHT = int(SCREEN_HEIGHT * 0.70)  # 504px
CONFIDENCE_AREA_HEIGHT = SCREEN_HEIGHT - GAME_AREA_HEIGHT  # 216px

# Colors
COLOR_BACKGROUND = (10, 10, 15)
COLOR_ORB = (100, 200, 255)
COLOR_TUNNEL_TOP = (40, 40, 60)
COLOR_TUNNEL_BOTTOM = (40, 40, 60)
COLOR_OBSTACLE = (200, 80, 80)
COLOR_COIN = (255, 215, 0)

# Gesture names (9 gestures from constants.py)
GESTURE_NAMES = [
    "index_press",
    "index_release",
    "middle_press",
    "middle_release",
    "thumb_click",
    "thumb_down",
    "thumb_in",
    "thumb_out",
    "thumb_up",
]

# Gesture display names (shorter for UI)
GESTURE_DISPLAY_NAMES = {
    "index_press": "index ↓",
    "index_release": "index ↑",
    "middle_press": "middle ↓",
    "middle_release": "middle ↑",
    "thumb_click": "thumb ●",
    "thumb_down": "thumb ↓",
    "thumb_in": "thumb ←",
    "thumb_out": "thumb →",
    "thumb_up": "thumb ↑",
}

# Power-ups triggered by confident gesture detection (>70% confidence)
POWERUP_CONFIGS = {
    "index_press": {
        "name": "Speed Boost",
        "color": (255, 50, 50),
        "duration": 3.0,
        "effect": "speed",
        "multiplier": 1.5,
    },
    "index_release": {
        "name": "Slow Motion",
        "color": (100, 150, 255),
        "duration": 4.0,
        "effect": "slow_mo",
        "multiplier": 0.5,
    },
    "middle_press": {
        "name": "Shield",
        "color": (50, 255, 50),
        "duration": 5.0,
        "effect": "shield",
        "multiplier": 1.0,
    },
    "middle_release": {
        "name": "Double Points",
        "color": (255, 215, 0),
        "duration": 5.0,
        "effect": "double_points",
        "multiplier": 2.0,
    },
    "thumb_click": {
        "name": "Magnet",
        "color": (200, 50, 255),
        "duration": 4.0,
        "effect": "magnet",
        "multiplier": 1.0,
    },
    "thumb_down": {
        "name": "Gravity Flip",
        "color": (255, 150, 0),
        "duration": 3.0,
        "effect": "gravity_flip",
        "multiplier": -1.0,
    },
    "thumb_in": {
        "name": "Shrink",
        "color": (255, 255, 255),
        "duration": 4.0,
        "effect": "shrink",
        "multiplier": 0.5,
    },
    "thumb_out": {
        "name": "Grow",
        "color": (255, 150, 50),
        "duration": 3.0,
        "effect": "grow",
        "multiplier": 2.0,
    },
    "thumb_up": {
        "name": "Flight Mode",
        "color": (50, 255, 255),
        "duration": 4.0,
        "effect": "flight",
        "multiplier": 1.0,
    },
}

# Keyboard mappings for simulation
KEYBOARD_MAP = {
    arcade.key.UP: "emg_increase",  # Simulate muscle activation
    arcade.key.DOWN: "emg_decrease",  # Simulate relaxation
    arcade.key.KEY_1: "index_press",
    arcade.key.KEY_2: "index_release",
    arcade.key.KEY_3: "middle_press",
    arcade.key.KEY_4: "middle_release",
    arcade.key.KEY_5: "thumb_click",
    arcade.key.KEY_6: "thumb_down",
    arcade.key.KEY_7: "thumb_in",
    arcade.key.KEY_8: "thumb_out",
    arcade.key.KEY_9: "thumb_up",
}

# Physics constants
ORB_RADIUS = 20
ORB_GRAVITY = 0.3  # Pixels per frame^2
ORB_LIFT_FORCE = 1.5  # Pixels per frame when EMG active
ORB_MAX_SPEED = 8.0

# Game constants
TUNNEL_SCROLL_SPEED = 3.0
OBSTACLE_SPAWN_INTERVAL = 2.0  # seconds
COIN_SPAWN_INTERVAL = 0.5  # seconds
CONFIDENCE_THRESHOLD_POWERUP = 0.70  # 70% confidence to trigger power-up

# Serial settings
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200
CONFIDENCE_THRESHOLD_DETECT = 0.5  # Match firmware threshold
