"""
Configuration Module for AI Traffic Control System
Contains all hyperparameters and constants
"""

# === ENVIRONMENT SETTINGS ===
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
DASHBOARD_WIDTH = 400
WINDOW_WIDTH = SCREEN_WIDTH + DASHBOARD_WIDTH
FPS = 60

# Traffic Configuration
LANE_WIDTH = 120
SAFE_DISTANCE = 80
MIN_GREEN_TIME = 200  # CHANGE THIS: Minimum frames before switch (higher = longer green)
MAX_GREEN_TIME = 450  # CHANGE THIS: Maximum frames for any green light
YELLOW_DURATION = 60  # Yellow light duration in frames

# Emergency vehicle settings
AMBULANCE_PRIORITY = True  # Enable immediate green for ambulances
AMBULANCE_OVERRIDE_MIN_TIME = True  # Allow switching before MIN_GREEN_TIME for ambulances

# IMPROVED SPAWN RATES - More balanced traffic
SPAWN_RATE = 0.055  # Increased base vehicle spawn probability per frame
AMBULANCE_SPAWN_RATE = 0.075  # Increased ambulance spawn probability
LANE_SPAWN_BALANCE = True  # Enable balanced spawning across lanes

# === RL HYPERPARAMETERS ===
STATE_SIZE = 16  # [4 car counts, 4 wait times, 4 lights, 4 ambulance flags]
ACTION_SIZE = 2  # 0: Keep current, 1: Switch to next
HIDDEN_SIZE = 128
BATCH_SIZE = 64  # Reduced for faster learning
GAMMA = 0.95  # Discount factor
LEARNING_RATE = 0.001  # Increased for faster convergence
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 300  # Update target network every N steps

# Exploration - Faster decay for quicker exploitation
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.993  # Faster decay

# Training
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 1500
SAVE_INTERVAL = 50  # Save model every N episodes

# === REWARD STRUCTURE - REBALANCED ===
REWARDS = {
    'per_waiting_car': -2.0,  # Increased penalty
    'switch_penalty': -5,  # Higher switching cost
    'ambulance_moving': 200,  # Huge reward for moving ambulance
    'ambulance_waiting': -150,  # Severe penalty for waiting ambulance
    'throughput_bonus': 3,  # Per car that exits
    'congestion_penalty': -1.0,  # Per car over threshold
    'empty_lane_penalty': -10  # Penalty for keeping empty lane green
}

CONGESTION_THRESHOLD = 6  # Cars per lane considered congested

# === VISUAL SETTINGS ===
COLORS = {
    'background': (25, 25, 35),
    'road': (45, 45, 55),
    'line': (255, 255, 255),
    'dash_bg': (20, 20, 28),
    'red': (220, 50, 50),
    'yellow': (255, 220, 0),
    'green': (50, 220, 100),
    'cyan': (0, 230, 230),
    'text': (240, 240, 245),
    'car': (70, 130, 255),
    'ambulance': (255, 60, 60)
}

# === FILE PATHS ===
MODEL_SAVE_PATH = "models/traffic_dqn.pth"
ANALYTICS_PATH = "analytics/"
LOG_PATH = "logs/training.log"