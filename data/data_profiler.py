import numpy as np


class DataProfiler:
    def profile(self, dataset):
        X = dataset.X_train
        y = dataset.y_train

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        missing_ratio = np.isnan(X).mean()

        class_counts = np.bincount(y)
        imbalance_ratio = class_counts.min() / class_counts.max()

        numeric_ratio = 1.0

        n_samples_norm = min(n_samples / 5000.0, 1.0)
        n_features_norm = min(n_features / 100.0, 1.0)
        n_classes_norm = min(n_classes / 20.0, 1.0)

        return {
            "n_samples_norm": n_samples_norm,
            "n_features_norm": n_features_norm,
            "n_classes_norm": n_classes_norm,
            "missing_ratio": float(missing_ratio),
            "imbalance_ratio": float(imbalance_ratio),
            "numeric_ratio": float(numeric_ratio),
        }
