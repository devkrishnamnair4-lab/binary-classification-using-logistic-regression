# Report — Logistic Regression Classification (Task 4)

## Objective
Build a binary classifier using logistic regression and evaluate it using confusion matrix, precision, recall, and ROC-AUC. Also demonstrate threshold tuning and explain the sigmoid function.

## Dataset
Breast Cancer Wisconsin dataset (from scikit-learn). Target: malignant (0) / benign (1).

## Preprocessing
- Train/test split with 80/20 split and stratification.
- Features standardized using `StandardScaler`.

## Model
- `sklearn.linear_model.LogisticRegression` (liblinear solver, max_iter=1000).

## Evaluation (example output when running `app.py`)
- Accuracy (threshold 0.5): *see console output*
- Precision (threshold 0.5): *see console output*
- Recall (threshold 0.5): *see console output*
- ROC-AUC: *see console output*
- Confusion matrices printed for threshold 0.5 and tuned threshold.

## Threshold tuning
We compute the ROC curve and choose the threshold that maximizes Youden's J statistic (tpr - fpr). This balances sensitivity and specificity.

## Sigmoid function
The logistic (sigmoid) function maps any real-valued input to (0,1):
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]
Logistic regression models the log-odds as a linear function and then applies the sigmoid to produce probabilities.

## Notes & Next steps
- You can further improve the pipeline by adding cross-validation, regularization tuning (C parameter), and feature selection.
- For imbalanced classes, consider class weighting, resampling, or alternative metrics (e.g., F1, PR-AUC).

Reference: task PDF provided by the user. fileciteturn0file0
