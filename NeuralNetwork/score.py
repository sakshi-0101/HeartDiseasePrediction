import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def score(y_test, test_predictions, test_probabilities):
    y_true = y_test.numpy().flatten()
    y_pred = test_predictions.numpy().flatten()
    y_prob = test_probabilities.numpy().flatten()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n")
    print("*" * 30)
    print("Neural Network Results")
    print("*" * 30)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
