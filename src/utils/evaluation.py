from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .comp_metric import pauc_score

def compute_metrics(y_true: list, y_prob: list) -> tuple[float, float, float, float]:
    y_pred = [1 if x > 0.5 else 0 for x in y_prob]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    pauc = pauc_score(y_true, y_prob, min_tpr=0.8)
    return acc, f1, auc, pauc