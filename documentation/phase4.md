# Phase 4: Modeling & Evaluation

## Phase Overview

This phase focused on building, training, and evaluating machine learning models to predict customer churn. It included:

- Preparing the feature-engineered dataset
- Splitting data into training and testing sets
- Training multiple classification models
- Evaluating model performance using various metrics
- Hyperparameter tuning of the best model (Gradient Boosting)
- Analyzing feature importance
- Saving the final model

---

## 1. Data Preparation for Modeling

- **Dataset Used**: `customer_churn_features.csv` (from Phase 3)
- **Target Variable**: `is_churned`
- **Features**: RFM, Tenure, one-hot encoded Country
- **Excluded**: `CustomerID` (identifier, not predictive)

### Splitting Data

- Train-Test Split:  
  - 75% training, 25% testing  
  - Stratified on `y` to preserve churn ratio  
  - `random_state` set for reproducibility

---

## 2. Model Selection and Initial Training

Three models were selected:

1. **Logistic Regression**  
   - Baseline model  
   - Used with `StandardScaler` in a `Pipeline`  
   - Sensitive to feature scaling

2. **Random Forest Classifier**  
   - Tree-based ensemble model  
   - Handles non-linear relationships  
   - No scaling needed

3. **Gradient Boosting Classifier**  
   - Sequential ensemble model  
   - Known for high accuracy  
   - No scaling needed

Each model was trained on the `X_train` and `y_train` sets.

---

## 3. Model Evaluation Metrics

Each model was evaluated on the `X_test` set using:

- **Accuracy**: Overall correctness
- **Precision**: Correctly predicted churners / Total predicted churners
- **Recall**: Correctly predicted churners / Total actual churners
- **F1-Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Discrimination ability across all thresholds
- **Confusion Matrix**: Breakdown of TP, FP, FN, TN

### Initial ROC-AUC Scores:

- Logistic Regression: **0.7949**
- Random Forest: **0.7818**
- Gradient Boosting: **0.8120**

> **Gradient Boosting** had the best ROC-AUC and was selected for tuning.

### Priority Metrics:

- **Recall** (to capture actual churners)
- **ROC-AUC** (for overall discriminative performance)

---

## 4. Hyperparameter Tuning (Gradient Boosting)

Used `GridSearchCV` to tune:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`

### Best Tuned Model Performance on Test Set:

- **Accuracy**: `0.7517`
- **Precision**: `0.7536`
- **Recall**: `0.8342`
- **F1-Score**: `0.7919`
- **ROC-AUC**: `0.8121`

### Confusion Matrix:

|                | Predicted Non-Churn | Predicted Churn |
|----------------|---------------------|------------------|
| **Actual Non-Churn** | 369 (TN)             | 204 (FP)         |
| **Actual Churn**     | 124 (FN)             | 624 (TP)         |

> Strong recall and ROC-AUC confirm the model is effective at identifying churners.

---

## 5. Feature Importance Analysis

Used `feature_importances_` from the tuned Gradient Boosting model.

- Features ranked by importance
- Top 15 visualized in a bar plot

> Example Insight:  
> If **Recency** is highly important, customers who haven’t purchased recently are more likely to churn.

This helps translate model findings into actionable business strategies.

---

## 6. Model Saving

The final tuned Gradient Boosting model was saved as:

- Saved using `joblib`
- Can be reloaded for inference or deployment