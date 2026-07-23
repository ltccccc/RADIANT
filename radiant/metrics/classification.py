import numpy as np


def binary_metrics(prob, target, threshold=0.5):
    prob = np.asarray(prob)
    target = np.asarray(target).astype(int)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    auc = roc_auc(prob, target)
    acc = (tp + tn) / max(len(target), 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)
    return {"auc": auc, "accuracy": acc, "sensitivity": sensitivity, "specificity": specificity, "f1": f1}


def roc_auc(prob, target):
    pos = prob[target == 1]
    neg = prob[target == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))

