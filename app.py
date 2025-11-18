#!/usr/bin/env python3
"""app.py
Train a logistic regression classifier on the Breast Cancer Wisconsin dataset,
evaluate it, and demonstrate threshold tuning.
"""
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    roc_auc_score, roc_curve, accuracy_score
)
import matplotlib.pyplot as plt

def main():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(X_train_s, y_train)

    # Predicted probabilities and default threshold 0.5
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds_05 = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds_05)
    prec = precision_score(y_test, preds_05)
    rec = recall_score(y_test, preds_05)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds_05)

    print("Evaluation with threshold = 0.5")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # ROC curve (saved to file)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    print("Saved ROC curve to 'roc_curve.png'")

    # Threshold tuning example: pick threshold that maximizes tpr - fpr (Youden's J)
    j_scores = tpr - fpr
    ix = np.argmax(j_scores)
    best_threshold = thresholds[ix]
    print(f"Suggested threshold (Youden's J): {best_threshold:.4f}")

    preds_best = (probs >= best_threshold).astype(int)
    prec_best = precision_score(y_test, preds_best)
    rec_best = recall_score(y_test, preds_best)
    cm_best = confusion_matrix(y_test, preds_best)
    print("Evaluation with tuned threshold:")
    print(f"Precision: {prec_best:.4f}")
    print(f"Recall: {rec_best:.4f}")
    print("Confusion Matrix:")
    print(cm_best)

    # Save model and scaler
    with open('model_and_scaler.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler, 'feature_names': feature_names.tolist()}, f)
    print("Saved trained model and scaler to 'model_and_scaler.pkl'")

if __name__ == '__main__':
    main()
