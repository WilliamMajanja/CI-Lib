"""Evaluation metrics for classification and regression."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def mse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float(1.0) if ss_res == 0 else float(0.0)
    return float(1.0 - ss_res / ss_tot)


def confusion_matrix(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike
) -> npt.NDArray[np.intp]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.intp)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def precision_recall_f1(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, average: str = "macro"
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = fp = fn = 0
        for c in classes:
            tp += int(np.sum((y_true == c) & (y_pred == c)))
            fp += int(np.sum((y_true != c) & (y_pred == c)))
            fn += int(np.sum((y_true == c) & (y_pred != c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return float(precision), float(recall), float(f1)

    if average == "macro":
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for c in classes:
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

    raise ValueError(f"Unknown average '{average}'. Use 'macro' or 'micro'.")
