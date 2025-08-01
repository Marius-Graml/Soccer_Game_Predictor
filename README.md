# ⚽ MSP: Macro-level Soccer Predictor

**MSP** is a probabilistic, interpretable framework for forecasting soccer match outcomes. Instead of relying on opaque deep learning models, this project leverages **probabilistic graphical models (PGMs)** with **latent variables** to generate structured, explainable predictions based on macro-level team attributes.

## 🧠 Project Overview

**Goal**: Predict the outcome of soccer matches (Home Win, Draw, Away Win) with an emphasis on **interpretability** rather than pure accuracy.

**Approach**:
- Use of a **Latent Bayesian Network (Latent BN)** that models unobserved _team strength_ variables derived from observable match features.
- Benchmark against traditional models: Logistic Regression, Random Forest, and XGBoost.
- Introduce a **Bayesian Flat-Logit** model for probabilistic baseline comparisons using continuous features.
- Dataset: Structured macro-level data from [Transfermarkt](https://www.transfermarkt.com/) covering player market values, club statistics, and match outcomes.

---

## 📊 Key Features

- **Interpretable Predictions**: Unlike black-box models, our framework enables users to trace predictions back to team-level characteristics.
- **Latent Variables**: Home and away team strength are inferred through unsupervised learning and serve as interpretable intermediates.
- **Modular Design**: Easily extensible for new features or competitions.
- **Trade-off Analysis**: Evaluate the balance between performance and explainability using balanced and imbalanced datasets.

---

## 🔍 Models Implemented

| Model                   | Type                  | Interpretability | Accuracy | 
|------------------------|-----------------------|------------------|----------|
| Latent Bayesian Network| Generative (PGM)      | ✅ High          | ⚠️ Medium| 
| Bayesian Flat-Logit    | Discriminative (Bayes)| ✅ Medium-High   | ✅ Good  | 
| Logistic Regression     | Classical ML          | ✅ High          | ⚠️ Medium| 
| Random Forest / XGBoost| Classical ML          | ❌ Low           | ✅ Good  |

---

## 📁 Dataset

- Source: [Transfermarkt (2025)](https://www.kaggle.com/datasets/davidcariboo/player-scores/data)
- Samples: 59,605 matches
- Features: 16 engineered variables grouped into market value, nationality ratios, average age, win rates, goal differentials, points per game, and rest days.
- Target: Match outcome (Home Win / Draw / Away Win)

---

## 🔬 Experimental Highlights

- **Soft-EM Algorithm** used for latent variable model learning.
- **KMeans Initialization** improves cluster stability and interpretability.
- **Feature Discretization & Clustering** provides semantically rich latent team strength levels (e.g., weak → elite).
- **Balanced vs. Unbalanced Training** reveals trade-offs between overall accuracy and class sensitivity (especially for draws).

---

## 📈 Results Snapshot

**Latent BN (Balanced)**:
- Accuracy: 40.7%
- Balanced precision across all classes
- Supports full interpretability from features → strength → prediction

**Flat-Logit (Unbalanced)**:
- Accuracy: 53.7%
- High precision on home win, weak on draws
- Quantifies uncertainty with full Bayesian posterior

---

## 📌 Key Takeaways

- **Interpretability Matters**: Especially for tactical decision-making, betting insights, or analytical applications.
- **Latent Variables are Powerful**: Serve as abstractions that bridge raw features and predictive outcomes.
- **Prediction ≠ Understanding**: Black-box models may outperform, but PGMs explain _why_.

---

## 🛠️ How to Run the Code

### 📦 1. Baseline Models (`baseline_models.py`)

Train and evaluate baseline models such as Logistic Regression, Random Forest, and XGBoost using engineered macro features.

#### 🔄 Example Usage

```bash
python models/baseline_models.py --dataset full --balanced True

| Argument     | Type | Options         | Description                           |
| ------------ | ---- | --------------- | ------------------------------------- |
| `--dataset`  | str  | `small`, `full` | Select the dataset version            |
| `--balanced` | bool | `True`, `False` | Whether to balance class distribution |


### 🔁 2. EM Latent Model (`em_ext.py`)

Train interpretable latent variable models using the Expectation-Maximization (EM) algorithm on macro-level match features. Supports both Hard-EM and Soft-EM.

#### 🔄 Example Usage

```bash
python models/em_ext.py --em-type soft --balanced True --init kmeans --bins 3 --iterations 5 --restarts 3 --dataset full

| Argument       | Type | Options                                | Description                                                        |
| -------------- | ---- | -------------------------------------- | ------------------------------------------------------------------ |
| `--em-type`    | str  | `hard`, `soft`                         | Select the EM variant: Hard-EM (greedy) or Soft-EM (probabilistic) |
| `--balanced`   | bool | `True`, `False`                        | Whether to use class-balanced training data                        |
| `--init`       | str  | `random`, `uniform`, `kmeans`, `multi` | Latent class initialization strategy                               |
| `--bins`       | int  | ≥ 2                                    | Number of latent clusters (e.g., team strength levels)             |
| `--iterations` | int  | ≥ 1                                    | Number of EM steps (only for Soft-EM)                              |
| `--restarts`   | int  | ≥ 1                                    | Number of model restarts for stability (Soft-EM only)              |
| `--dataset`    | str  | `small`, `full`                        | Choose the dataset variant                                         |

---

## 📎 Related Links

- [Project Report (PDF)](./Project_Report.pdf)
- [Transfermarkt Dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores/data)

