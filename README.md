# 🧱 ML Foundations

The engineering workflow behind machine learning — not just the math, but how projects are actually built end to end.

Three projects covering the full supervised learning pipeline: backpropagation implemented by hand, classical ML on tabular data, a CNN built from scratch in PyTorch and professional experiment tracking with automated hyperparameter search.

**Status: Complete ✅**

## Projects

### 1. No-Magic Neural Net 🔬
*Backpropagation by hand, in pure NumPy*

A 3-layer neural network built with zero ML frameworks — forward pass, chain rule, backprop, and gradient descent all implemented manually. Applied to binary image classification.

**Stack:** NumPy · matplotlib

Key concepts: the chain rule as backprop, why autograd exists, manual gradient computation, weight initialization from first principles.

---

### 2. Titanic Oracle 🚢
*Survival prediction on the most iconic ML dataset*

The full supervised learning workflow on structured data — EDA, feature engineering, sklearn pipelines, cross-validated model comparison across 7 algorithms, Bayesian hyperparameter tuning with Optuna, and SHAP explainability.

**Result:** 0.758 on the Kaggle leaderboard (vs ~83% cross-validated accuracy — the generalization gap is the main lesson)
**Stack:** pandas · scikit-learn · XGBoost · Optuna · SHAP · seaborn

Key concepts: data leakage and how pipelines prevent it, stratified k-fold cross-validation, group imputation vs global imputation, why feature engineering beats algorithm selection.

---


### 3. Cat vs Dog Classifier + Mission Control 🐱🐕🎛️
*A CNN built from scratch, then tracked, swept, and benchmarked like a professional*

Three phases building on each other:

**Built a CNN from the ground up in PyTorch** — 3 blocks (Conv2d → BatchNorm → ReLU → MaxPool), a custom `Dataset` class wrapping HuggingFace's cats_vs_dogs (23,410 images), a hand-written training loop, and data augmentation.

**Wired in Weights & Biases** — automatic logging of loss curves, gradient histograms, learning rate schedules, sample predictions with confidence scores, and versioned model artifacts. Then a 20-trial Bayesian hyperparameter sweep across learning rate, filter count, dropout, batch size, and weight decay.

**Benchmarked against transfer learning** — a frozen pretrained ResNet-18 with a replaced classification head, compared head-to-head against the scratch CNN in the same dashboard.

| Model | Total params | Trainable params | Test accuracy |
|---|---|---|---|
| Scratch CNN | ~9,537,346 | ~9,537,346 | 91.41% |
| ResNet-18 (frozen) | ~11,177,538 | ~1,026 | 97.69% |

The ResNet is a slightly larger network overall, but it trains roughly 9,000× fewer parameters and still wins by 6 points — transfer learning in one experiment.

**Stack:** PyTorch · torchvision · Weights & Biases · HuggingFace datasets

Key concepts: `Dataset` / `DataLoader` internals, the 4-step training pattern, `model.train()` vs `model.eval()` and why BatchNorm behaves differently in each, `torch.no_grad()`, Bayesian hyperparameter search, model artifacts and versioning, why pretrained models need ImageNet normalization values, test set discipline.


---

## What this repo demonstrates

- Building a supervised learning pipeline end to end, from raw data to submission
- Implementing backpropagation manually before using a framework that hides it
- Writing custom PyTorch `Dataset` classes and training loops rather than using high-level wrappers
- Tracking, comparing, and reproducing experiments like a professional
- Understanding *why* each design decision was made — every project includes written notes on the reasoning and the bugs encountered

---

*Projects 1.1–1.3 of an ongoing roadmap building AI projects from the ground up — see the [hub repo](https://github.com/Altster/AI-Projects) for the full roadmap*
