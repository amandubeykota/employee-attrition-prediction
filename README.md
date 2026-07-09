# employee-attrition-prediction
Predicting employee turnover with Logistic Regression, Decision Tree, and Random Forest — EDA to model comparison to HR recommendations.

# Employee Attrition Prediction

An end-to-end machine learning project predicting employee turnover for **Salifort Motors** (a fictional case-study company). The project moves through the full analytics lifecycle — data cleaning, exploratory analysis, three classification models compared under cross-validated grid search, and model results translated into stakeholder-ready HR recommendations.

## Business Problem

Replacing an employee is expensive. Salifort Motors wants to understand *why* employees leave and identify those at risk of leaving **before** they disengage, so HR can intervene with workload, recognition, or career-development changes.

## Dataset

`HR_capstone_dataset.csv` — ~15,000 employee-level records including:

- Satisfaction level and last evaluation score
- Number of projects and average monthly hours
- Tenure, work accidents, promotions in the last 5 years
- Department and salary band
- Target: `left` (whether the employee left the company)

## Methodology

1. **Cleaning** — column standardization, missing-value check, removal of duplicate rows, IQR-based outlier detection on tenure.
2. **EDA** — turnover rate; workload vs project count for leavers vs stayers; hours vs satisfaction; satisfaction by tenure; correlation heatmap; attrition by department. The EDA surfaces distinct leaver profiles (e.g., overworked high-performers and under-utilized low-satisfaction employees).
3. **Modeling** — supervised binary classification with three models:
   - Logistic Regression (baseline, trained on the outlier-removed data)
   - Decision Tree with `GridSearchCV` (4-fold CV, tuned on ROC AUC)
   - Random Forest with `GridSearchCV` (4-fold CV, tuned on ROC AUC)
4. **Evaluation** — precision, recall, F1, accuracy, and AUC on a stratified 25% hold-out test set; trained models persisted with pickle.
5. **Interpretation** — Random Forest feature importance, explicitly framed as model weighting rather than proof of causation.

## Results

| Model | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| Logistic Regression* | 0.4667 | 0.2378 | 0.3150 | 0.8256 | 0.8923 |
| Decision Tree | 0.9362 | 0.9137 | 0.9248 | 0.9753 | 0.9780 |
| **Random Forest** | **0.9642** | **0.9197** | **0.9414** | **0.9810** | **0.9846** |

\* Trained and evaluated on the outlier-removed dataset.

The **Random Forest** is the recommended model, with the strongest performance across every metric. The most influential features align with the EDA: satisfaction level, evaluation results, project load, monthly hours, and tenure.

## Recommendations

- Monitor satisfaction, project load, monthly hours, tenure, and recent evaluation patterns as early-warning signals.
- Review workload balance and cap simultaneous project assignments.
- Strengthen recognition and career-growth conversations for at-risk groups.
- Validate the model on newer employee data before operational use, and incorporate qualitative signals (surveys, manager notes).
- **Ethical use:** predictions should support employee well-being and fair treatment — a decision-support tool, never the sole basis for HR action.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook Salifort_Motors_Employee_Retention_Analysis.ipynb
```

Place the CSV alongside the notebook and run all cells top to bottom. Note: the Random Forest grid search takes several minutes. The notebook writes `log_clf.pkl` and `rf1.pkl` — these are regenerated on each run and are excluded via `.gitignore`.

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn
