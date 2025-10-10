#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

def pr_auc(y_true, y_prob):
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return float("nan")

def roc_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return float("nan")

def best_f1_from_probs(y_true, y_prob):
    ps, rs, ths = precision_recall_curve(y_true, y_prob)
    f1s = 2*ps*rs/(ps+rs+1e-9)
    best_idx = int(np.nanargmax(f1s))
    return float(f1s[best_idx]), float(ths[max(0, best_idx-1)])
