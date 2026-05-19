"""
Tests for DatasetManager and DataProfiler.

These tests verify that the data loading and profiling modules are functional
without touching any RL, DQN, evaluator, or tool-execution components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_profiler import DataProfiler
from src.data.dataset_manager import DatasetManager


REQUIRED_DATASET_KEYS = {
    "name",
    "X",
    "y",
    "X_train",
    "X_val",
    "X_test",
    "y_train",
    "y_val",
    "y_test",
    "feature_names",
    "target_names",
    "n_samples",
    "n_features",
    "n_classes",
}

REQUIRED_PROFILE_KEYS = {
    "n_samples",
    "n_features",
    "n_classes",
    "missing_ratio",
    "class_imbalance_ratio",
    "numeric_ratio",
    "mean_abs_correlation",
}


def test_list_datasets_returns_supported_datasets() -> None:
    """DatasetManager should report all supported built-in datasets."""
    assert DatasetManager.list_datasets() == [
        "iris",
        "wine",
        "breast_cancer",
        "digits",
    ]


def test_load_dataset_returns_required_keys() -> None:
    """Loading iris should return the expected dataset dictionary structure."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    assert REQUIRED_DATASET_KEYS.issubset(dataset.keys())


def test_splits_are_non_empty() -> None:
    """Train, validation, and test splits should all contain samples."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    assert dataset["X_train"].shape[0] > 0
    assert dataset["X_val"].shape[0] > 0
    assert dataset["X_test"].shape[0] > 0
    assert dataset["y_train"].shape[0] > 0
    assert dataset["y_val"].shape[0] > 0
    assert dataset["y_test"].shape[0] > 0


def test_profile_returns_required_meta_features() -> None:
    """DataProfiler.profile should return all required meta-feature keys."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    profile = DataProfiler().profile(dataset["X"], dataset["y"])
    assert REQUIRED_PROFILE_KEYS.issubset(profile.keys())


def test_to_vector_returns_expected_shape() -> None:
    """DataProfiler.to_vector should return a seven-value float32 vector."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    profiler = DataProfiler()
    profile = profiler.profile(dataset["X"], dataset["y"])
    vector = profiler.to_vector(profile)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (7,)
    assert vector.dtype == np.float32


def test_load_all_supported_datasets_does_not_crash() -> None:
    """All supported datasets should load and profile successfully."""
    manager = DatasetManager(random_state=42)
    profiler = DataProfiler()

    datasets = manager.load_all()
    assert set(datasets) == set(manager.list_datasets())

    for dataset in datasets.values():
        profile = profiler.profile_dataset(dataset)
        vector = profiler.to_vector(profile)
        assert vector.shape == (7,)
