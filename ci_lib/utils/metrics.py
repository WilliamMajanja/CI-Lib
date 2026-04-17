"""Evaluation metrics for classification and regression."""
import numpy as np


def accuracy(y_true, y_pred):
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : array_like
        Ground-truth labels.
    y_pred : array_like
        Predicted labels.

    Returns
    -------
    float
        Fraction of correctly classified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def mse(y_true, y_pred):
    """Compute mean squared error.

    Parameters
    ----------
    y_true : array_like
        Ground-truth values.
    y_pred : array_like
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """Compute root mean squared error.

    Parameters
    ----------
    y_true : array_like
        Ground-truth values.
    y_pred : array_like
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    """Compute mean absolute error.

    Parameters
    ----------
    y_true : array_like
        Ground-truth values.
    y_pred : array_like
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true, y_pred):
    """Compute the coefficient of determination (R²).

    Parameters
    ----------
    y_true : array_like
        Ground-truth values.
    y_pred : array_like
        Predicted values.

    Returns
    -------
    float
        R² score.  1.0 indicates a perfect fit.

    Notes
    -----
    When the total sum of squares is zero (constant *y_true*), the function
    returns 1.0 if predictions are perfect and 0.0 otherwise, following the
    scikit-learn convention.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float(1.0) if ss_res == 0 else float(0.0)
    return float(1.0 - ss_res / ss_tot)


def confusion_matrix(y_true, y_pred):
    """Compute a confusion matrix.

    Parameters
    ----------
    y_true : array_like
        Ground-truth labels.
    y_pred : array_like
        Predicted labels.

    Returns
    -------
    numpy.ndarray
        2-D integer array of shape ``(n_classes, n_classes)`` where entry
        ``[i, j]`` is the count of samples with true label *i* predicted
        as label *j*.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.intp)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def precision_recall_f1(y_true, y_pred, average="macro"):
    """Compute precision, recall and F1 score.

    Parameters
    ----------
    y_true : array_like
        Ground-truth labels.
    y_pred : array_like
        Predicted labels.
    average : {'macro', 'micro'}, default='macro'
        ``'macro'`` computes per-class metrics and averages them.
        ``'micro'`` computes metrics globally from total TP / FP / FN.

    Returns
    -------
    precision : float
    recall : float
    f1 : float

    Raises
    ------
    ValueError
        If *average* is not ``'macro'`` or ``'micro'``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = 0
        fp = 0
        fn = 0
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
        precisions = []
        recalls = []
        f1s = []
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
        return (
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1s)),
        )

    raise ValueError(f"Unknown average '{average}'. Use 'macro' or 'micro'.")
