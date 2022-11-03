import numpy as np
from sklearn import metrics


def compute_metrics(y_true, y_score):
    y_pred = y_score.argmax(-1)
    return {
        'acc': metrics.accuracy_score(y_true, y_pred),
        'acc_top2': metrics.top_k_accuracy_score(y_true, y_score, k=2),
        'precision': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    }