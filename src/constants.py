"""
Project-wide constants for the DQN Tool Selection Agent.

This module defines the initial discrete action space used by the tool
selection environment.
"""

ACTIONS = {
    0: "do_nothing",
    1: "standard_scaler",
    2: "minmax_scaler",
    3: "pca",
    4: "feature_selection",
    5: "random_forest",
    6: "svm",
    7: "knn",
    8: "evaluate",
}

PREPROCESS_ACTIONS = {1, 2, 3, 4}
MODEL_ACTIONS = {5, 6, 7}
TERMINAL_ACTION = 8
