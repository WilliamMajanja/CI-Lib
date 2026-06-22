"""Utility functions module."""
from ci_lib.utils.preprocessing import normalize, train_test_split, one_hot_encode, label_encode
from ci_lib.utils.metrics import accuracy, mse, rmse, mae, r_squared, confusion_matrix, precision_recall_f1
from ci_lib.utils.benchmarks import sphere, rosenbrock, rastrigin, ackley, griewank, schwefel

__all__ = [
    "normalize", "train_test_split", "one_hot_encode", "label_encode",
    "accuracy", "mse", "rmse", "mae", "r_squared", "confusion_matrix", "precision_recall_f1",
    "sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel",
]
