import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def obtain_metrics(y_true, y_pred_prob):
    y_pred = y_pred_prob.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    
    best_loss_metrics = {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": rec,
    }

    return best_loss_metrics
