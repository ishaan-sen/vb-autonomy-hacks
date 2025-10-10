import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def macro_report(y_true, y_pred, class_names):
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return rep, cm
