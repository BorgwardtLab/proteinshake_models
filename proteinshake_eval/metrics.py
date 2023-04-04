import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr


# def compute_metrics(y_true, y_score, task_type='classification, multi-class'):
#     if 'classification' in task_type:
#         if 'multi-class' in task_type:
#             y_pred = y_score.argmax(-1)
#             return {
#                 'acc': metrics.accuracy_score(y_true, y_pred),
#                 'acc_top2': metrics.top_k_accuracy_score(y_true, y_score, k=2, labels=np.arange(y_score.shape[1])),
#                 'precision': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
#                 'recall': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
#             }
#         elif 'binary' in task_type:
#             y_pred = (y_score > 0).astype('float32')
#             return {
#                 'acc': metrics.accuracy_score(y_true, y_pred),
#                 'auc': metrics.roc_auc_score(y_true, y_score),
#                 'aupr': metrics.average_precision_score(y_true, y_score),
#                 'mcc': metrics.matthews_corrcoef(y_true, y_pred),
#             }
#     elif 'regression' in task_type:
#         return {
#             'neg_mse': -metrics.mean_squared_error(y_true, y_score),
#             'neg_mae': -metrics.mean_absolute_error(y_true, y_score),
#             'mse': metrics.mean_squared_error(y_true, y_score),
#             'mae': metrics.mean_absolute_error(y_true, y_score),
#             'r2': metrics.r2_score(y_true, y_score)
#         }
#     else:
#         raise ValueError("Unknown task type!")


def compute_metrics(y_true, y_score, task):
    _, task_type = task.task_type
    y_pred = y_score
    if task_type == "multi_class":
        y_pred = y_score.argmax(-1)
        scores = task.evaluate(y_true, y_pred)
    elif task_type == "multi_label":
        y_pred = (y_score > 0).astype('float32')
        scores = task.evaluate(y_true, y_pred)
    elif task_type == "binary":
        y_pred = (y_score > 0).astype('float32')
        scores = task.evaluate(y_true, y_pred)
        scores['auc'] = metrics.roc_auc_score(y_true, y_score)
    elif task_type == 'regression':
        scores = task.evaluate(y_true, y_pred)
        scores['neg_mse'] = -scores['mse']
        scores['spearmanr'] = spearmanr(y_true, y_pred).correlation
        scores['r2'] = metrics.r2_score(y_true, y_score)
    else:
        scores = task.evaluate(y_true, y_pred)
    return scores
