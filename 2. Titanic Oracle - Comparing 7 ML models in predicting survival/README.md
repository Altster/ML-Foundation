# Titanic Oracle 🚢
Predicting Titanic passenger survival using classical machine learning — full pipeline from raw historical data to Kaggle submission.

**Kaggle score: 0.758** &nbsp;·&nbsp; **CV accuracy: 83.5%**



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

### The honest result: a generalization gap

Cross-validation said 83.5%. The Kaggle leaderboard said 75.8% — slightly *below* Kaggle's own one-rule baseline (predict that every woman survives and every man dies), which scores about 76.6%.

That gap is the most useful thing this project taught me. With only 891 training rows, a tuned XGBoost plus several engineered features had enough freedom to fit patterns that don't hold on the test passengers. And because the model was picked (and tuned) using the same CV folds, the winning CV score is an optimistic estimate by construction. A simpler model, fewer engineered features, or nested cross-validation would likely close part of the gap.



## Features engineered

| Feature | Source | Reasoning |
|---|---|---|
| `Title` | Name column | Encodes sex, age range, and social class in one variable |
| `FamilySize` | SibSp + Parch + 1 | Solo travellers and large groups both survived less |
| `IsAlone` | FamilySize | Binary flag for solo travellers |
| `FamilySizeType` | FamilySize | Buckets: alone / small / large |
| `AgeGroup` | Age | Buckets: child / teen / adult / middle / senior |
| `Embarked` | Embarked | 2 missing values filled with most frequent port |


## Tech stack

- **Python** — pandas, numpy
- **Machine learning** — scikit-learn, XGBoost
- **Hyperparameter tuning** — Optuna
- **Explainability** — SHAP
- **Visualisation** — matplotlib, seaborn

## Concepts covered

- Exploratory data analysis
- Feature engineering from raw columns
- Sklearn `Pipeline` and `ColumnTransformer`
- Data leakage and how pipelines prevent it
- Stratified k-fold cross-validation
- Decision Trees, Random Forests, SVMs, Gradient Boosting, XGBoost
- Bayesian hyperparameter search with Optuna
- SHAP explainability — global importance, beeswarm, and per-passenger force plots
