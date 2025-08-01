import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=["small", "full"], default="small", help="Which dataset to use"
)
parser.add_argument(
    "--balanced", type=bool, default=True, help="Enable class balancing"
)
args = parser.parse_args()

# === 1. Load dataset ===
if args.dataset == "small":
    data = pd.read_csv("data/drafts/small_dataset.csv")
else:
    data = pd.read_csv("data/dataset.csv")

# === 2. Prepare target variable ===
data = data[data["result"].isin(["Home Win", "Draw", "Away Win"])]
data["MatchOutcome"] = data["result"].map({"Home Win": 0, "Draw": 1, "Away Win": 2})

# === 3. Balance classes if enabled ===
print("\nDistribution before balancing:")
print(data["MatchOutcome"].value_counts())

if args.balanced:
    min_class_size = data["MatchOutcome"].value_counts().min()
    data = (
        data.groupby("MatchOutcome")
        .sample(n=min_class_size, random_state=42)
        .reset_index(drop=True)
    )

print("\nDistribution after balancing:")
print(data["MatchOutcome"].value_counts())

# === 4. Select all numeric features except target-related ones ===
excluded_columns = ["result", "MatchOutcome", "game_id"]
numeric_data = data.select_dtypes(include=[np.number])
features = [col for col in numeric_data.columns if col not in excluded_columns]

X = data[features]
y = data["MatchOutcome"]

# === 5. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 6. Feature scaling (for LR, XGB) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 7. Train and evaluate models ===
reports = {}

# Logistic Regression
lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
reports["Logistic Regression"] = classification_report(
    y_test, y_pred_lr, output_dict=True
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
reports["Random Forest"] = classification_report(y_test, y_pred_rf, output_dict=True)

# XGBoost
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
reports["XGBoost"] = classification_report(y_test, y_pred_xgb, output_dict=True)

# === 8. Create comparison table ===
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
print("\n=== Model Comparison ===")
print(summary_df)
