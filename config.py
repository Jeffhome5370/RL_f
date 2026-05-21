import os

SEED = 42

RESULT_DIR = "results"
LOG_DIR = os.path.join(RESULT_DIR, "logs")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")

ACTIONS = [
    "standard_scaler",
    "minmax_scaler",
    "pca",
    "feature_selection",
    "random_forest",
    "svm",
    "knn",
    "evaluate",
]

ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

MAX_STEPS = 5
INVALID_ACTION_PENALTY = -0.2
STEP_PENALTY = -0.05
PIPELINE_LENGTH_PENALTY = 0.02
INVALID_COUNT_PENALTY = 0.05

STATE_DIM = 12
ACTION_DIM = len(ACTIONS)

EPISODES = 300
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 200
TARGET_UPDATE_FREQ = 20

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
