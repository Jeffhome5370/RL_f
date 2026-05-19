"""
Demo script for DatasetManager and DataProfiler.

This script loads each supported built-in dataset, prints split shapes, and
shows the computed dataset profile and vector representation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_profiler import DataProfiler
from src.data.dataset_manager import DatasetManager


def main() -> None:
    """Run a small data loading and profiling demo."""
    manager = DatasetManager(random_state=42)
    profiler = DataProfiler()

    dataset_names = manager.list_datasets()
    print(f"Supported datasets: {dataset_names}")

    for name in dataset_names:
        dataset = manager.load_dataset(name)
        profile = profiler.profile_dataset(dataset)
        vector = profiler.to_vector(profile)

        print(f"\nDataset: {name}")
        print(f"Train shape: {dataset['X_train'].shape}, {dataset['y_train'].shape}")
        print(f"Validation shape: {dataset['X_val'].shape}, {dataset['y_val'].shape}")
        print(f"Test shape: {dataset['X_test'].shape}, {dataset['y_test'].shape}")
        print("Profile:")
        pprint(profile)
        print(f"Profile vector shape: {vector.shape}")
        print(f"Profile vector values: {vector}")


if __name__ == "__main__":
    main()
