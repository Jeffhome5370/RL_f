from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class DatasetManager:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def load_all(self) -> List[DatasetBundle]:
        datasets = []

        dataset_loaders = [
            ("iris", load_iris),
            ("wine", load_wine),
            ("breast_cancer", load_breast_cancer),
            ("digits", load_digits),
        ]

        for name, loader in dataset_loaders:
            data = loader()
            X = data.data.astype(np.float32)
            y = data.target

            bundle = self._split_dataset(name, X, y)
            datasets.append(bundle)

        return datasets

    def _split_dataset(self, name: str, X, y) -> DatasetBundle:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=self.seed,
            stratify=y_train_val,
        )

        return DatasetBundle(
            name=name,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
