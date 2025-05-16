import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load real data
data = pd.read_csv("final_dataset.csv")

# convert target variable to numerical data
data["result_code"] = data["result"].map({"Home Win": 0, "Draw": 1, "Away Win": 2})
data["MatchOutcome"] = data["result_code"]

# Check class distribution
print("\nClass distribution before balancing:")
print(data["MatchOutcome"].value_counts())

# Balance number of classes (equally distributed)
min_class_size = data["MatchOutcome"].value_counts().min()
balanced_data = (
    data.groupby("MatchOutcome")
    .sample(n=min_class_size, random_state=42)
    .reset_index(drop=True)
)

# Use balanced dataset from now
data = balanced_data

print("\nClass distribution after balancing:")
print(data["MatchOutcome"].value_counts())

# Features
features = [
    "home_avg_market_value",
    "home_nationalities",
    "home_avg_age",
    "home_total_minutes",
    "away_avg_market_value",
    "away_nationalities",
    "away_avg_age",
    "away_total_minutes",
]
X = data[features]
y = data["MatchOutcome"]

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardisierung für logistische Regression und XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ergebnisse speichern
reports = {}

# 1. Logistic Regression
lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
reports["Logistic Regression"] = classification_report(
    y_test, y_pred_lr, output_dict=True
)

# 2. Random Forest (kein Scaling nötig)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
reports["Random Forest"] = classification_report(y_test, y_pred_rf, output_dict=True)

# 3. XGBoost
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
reports["XGBoost"] = classification_report(y_test, y_pred_xgb, output_dict=True)

# Vergleichstabelle erstellen
summary = {
    model: {
        "accuracy": reports[model]["accuracy"],
        "macro_f1": reports[model]["macro avg"]["f1-score"],
        "weighted_f1": reports[model]["weighted avg"]["f1-score"],
        "precision_0": reports[model]["0"]["precision"],
        "recall_0": reports[model]["0"]["recall"],
        "precision_1": reports[model]["1"]["precision"],
        "recall_1": reports[model]["1"]["recall"],
        "precision_2": reports[model]["2"]["precision"],
        "recall_2": reports[model]["2"]["recall"],
    }
    for model in reports
}
summary_df = pd.DataFrame(summary).T

print(summary_df)
