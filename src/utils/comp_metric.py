import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def pauc_score(y_true: list, y_prob: list, min_tpr: float = 0.8):
    """
    Compute the partial AUC score
    """
    
    v_true = abs(np.array(y_true)-1)
    v_pred = np.array([1.0 - x for x in y_prob])
    max_fpr = abs(1 - min_tpr)
    pauc_scaled = roc_auc_score(v_true, v_pred, max_fpr=max_fpr)
    pauc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (pauc_scaled - 0.5)

    return pauc