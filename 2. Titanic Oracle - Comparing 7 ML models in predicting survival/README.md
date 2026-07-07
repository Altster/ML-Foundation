# Titanic Oracle 🚢
Predicting Titanic passenger survival using classical machine learning — full pipeline from raw historical data to Kaggle submission.

**Kaggle score: 0.758** &nbsp;·&nbsp; Top ~25% of all submissions



## Overview

Trains and compares 7 classical ML algorithms on the Titanic passenger dataset. Covers the complete supervised learning workflow: exploratory data analysis, feature engineering, sklearn preprocessing pipelines, cross-validated model comparison, Bayesian hyperparameter tuning with Optuna, and SHAP explainability.



## Results

| Model | CV Accuracy |
|---|---|
| XGBoost ✓ | 83.5% |
| SVM | 83.1% |
| Random Forest | 82.9% |
| Logistic Regression | 82.7% |
| Gradient Boosting | 81.9% |
| Decision Tree | 81.4% |
| KNN | 79.7% |

Baseline (predict all died): 61.6%

XGBoost selected for final submission. Hyperparameters tuned with Optuna (50 trials, TPE sampler).



## Features engineered

| Feature | Source | Reasoning |
|---|---|---|
| `Title` | Name column | Encodes sex, age range, and social class in one variable |
| `FamilySize` | SibSp + Parch + 1 | Solo travellers and large groups both survived less |
| `IsAlone` | FamilySize | Binary flag for solo travellers |
| `FamilySizeType` | FamilySize | Buckets: alone / small / large |
| `AgeGroup` | Age | Buckets: child / teen / adult / middle / senior |
| `Embarked` | Embarked | 2 missing values filled with most frequent port |

Age imputed using median per (Title × Pclass) group before bucketing — more accurate than global median imputation.



## Tech stack

- **Python** — pandas, numpy
- **Machine learning** — scikit-learn, XGBoost
- **Hyperparameter tuning** — Optuna
- **Explainability** — SHAP
- **Visualisation** — matplotlib, seaborn



## Project structure

```
titanic-oracle/
├── datasets/
│   ├── train.csv
│   └── test.csv
├── main_code.ipynb         ← full notebook
├── submissions.csv         ← Kaggle submission
├── model_CVcomparison.png  ← 7-model comparison chart
├── shap_importance.png     ← SHAP global feature importance
├── shap_beeswarm.png       ← SHAP beeswarm plot
├── shap_passenger.png      ← SHAP single passenger explanation
└── README.md
```


## Concepts covered

- Exploratory data analysis
- Feature engineering from raw columns
- Sklearn `Pipeline` and `ColumnTransformer`
- Data leakage and how pipelines prevent it
- Stratified k-fold cross-validation
- Decision Trees, Random Forests, SVMs, Gradient Boosting, XGBoost
- Bayesian hyperparameter search with Optuna
- SHAP explainability — global importance, beeswarm, and per-passenger force plots
