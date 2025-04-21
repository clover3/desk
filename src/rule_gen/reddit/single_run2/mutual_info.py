import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_mi_from_confusion(tp, fp, fn, tn):
    x, y = confusion_to_xy(tp, fp, fn, tn)
    importances = mutual_info_classif(x, y)
    return importances


def confusion_to_xy(tp, fp, fn, tn):
    xy = [(1, 1)] * tp + [(1, 0)] * fp + [(0, 1)] * fn + [(0, 0)] * tn
    x, y = zip(*xy)
    x = np.array(x)
    x = np.reshape(x, [-1, 1])
    return x, np.array(y)


def main():
    #19      4	25      40
    tp, fp, fn, tn = 19, 4, 25, 40
    importances = compute_mi_from_confusion(tp, fp, fn, tn)
    print(importances)
    importances = compute_mi_from_confusion(30, 35, 14, 9)
    print(importances)


if __name__ == "__main__":
    main()