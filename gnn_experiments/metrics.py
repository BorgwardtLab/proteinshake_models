import numpy as np
from sklearn import metrics


def compute_metrics(y_true, y_score, task_type='classification, multi-class'):
    if 'classification' in task_type:
        if 'multi-class' in task_type:
            y_pred = y_score.argmax(-1)
            return {
                'acc': metrics.accuracy_score(y_true, y_pred),
                'acc_top2': metrics.top_k_accuracy_score(y_true, y_score, k=2),
                'precision': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
            }
    elif 'regression' in task_type:
        return {
            'neg_mse': -metrics.mean_squared_error(y_true, y_score),
            'neg_mae': -metrics.mean_absolute_error(y_true, y_score),
            'mse': metrics.mean_squared_error(y_true, y_score),
            'mae': metrics.mean_absolute_error(y_true, y_score),
            'r2': metrics.r2_score(y_true, y_score)
        }
    else:
        raise ValueError("Unknown task type!")
