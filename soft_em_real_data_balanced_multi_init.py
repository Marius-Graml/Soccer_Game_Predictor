# Re-execute after environment reset to evaluate multiple Soft-EM initializations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report
from collections import defaultdict

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

# Parameter
features_home = [
    "home_avg_market_value",
    "home_nationalities",
    "home_avg_age",
    "home_total_minutes",
]
features_away = [
    "away_avg_market_value",
    "away_nationalities",
    "away_avg_age",
    "away_total_minutes",
]
__BINS = 3
n_iterations = 5
n_restarts = 3

# Diskretisieren
kbins_home = KBinsDiscretizer(n_bins=__BINS, encode="ordinal", strategy="uniform")
kbins_away = KBinsDiscretizer(n_bins=__BINS, encode="ordinal", strategy="uniform")
data[features_home] = kbins_home.fit_transform(data[features_home]).astype(int)
data[features_away] = kbins_away.fit_transform(data[features_away]).astype(int)

# Strukturdefinition
structure = [
    ("home_avg_market_value", "HomeStrength"),
    ("home_nationalities", "HomeStrength"),
    ("home_avg_age", "HomeStrength"),
    ("home_total_minutes", "HomeStrength"),
    ("away_avg_market_value", "AwayStrength"),
    ("away_nationalities", "AwayStrength"),
    ("away_avg_age", "AwayStrength"),
    ("away_total_minutes", "AwayStrength"),
    ("HomeStrength", "MatchOutcome"),
    ("AwayStrength", "MatchOutcome"),
]

# Ergebnisse für alle Durchläufe
all_results = []

# Mehrfache Initialisierungen
for restart in range(n_restarts):
    print("Restart " + str(restart + 1))
    train_data, test_data = train_test_split(
        data.copy(), test_size=0.2, random_state=42 + restart
    )

    # Zufällige Initialisierung
    train_data["HomeStrength"] = np.random.randint(0, __BINS, len(train_data))
    train_data["AwayStrength"] = np.random.randint(0, __BINS, len(train_data))
    for i in range(__BINS):
        train_data[f"HomeStrength_{i}"] = (train_data["HomeStrength"] == i).astype(
            float
        )
        train_data[f"AwayStrength_{i}"] = (train_data["AwayStrength"] == i).astype(
            float
        )

    # Soft-EM
    for iteration in range(n_iterations):
        print("Iteration" + str(iteration + 1))
        train_data["HomeStrength"] = train_data[
            [f"HomeStrength_{i}" for i in range(__BINS)]
        ].values.argmax(axis=1)
        train_data["AwayStrength"] = train_data[
            [f"AwayStrength_{i}" for i in range(__BINS)]
        ].values.argmax(axis=1)

        model = BayesianNetwork(structure)
        model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
        infer = VariableElimination(model)

        for idx, row in train_data.iterrows():
            evidence_home = {f: int(row[f]) for f in features_home}
            evidence_away = {f: int(row[f]) for f in features_away}
            try:
                home_post = infer.query(["HomeStrength"], evidence=evidence_home)
                away_post = infer.query(["AwayStrength"], evidence=evidence_away)
                for s in range(__BINS):
                    train_data.at[idx, f"HomeStrength_{s}"] = home_post.values[s]
                    train_data.at[idx, f"AwayStrength_{s}"] = away_post.values[s]
            except:
                continue

    # Finales Modell
    train_data["HomeStrength"] = train_data[
        [f"HomeStrength_{i}" for i in range(__BINS)]
    ].values.argmax(axis=1)
    train_data["AwayStrength"] = train_data[
        [f"AwayStrength_{i}" for i in range(__BINS)]
    ].values.argmax(axis=1)
    final_model = BayesianNetwork(structure)
    final_model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
    infer = VariableElimination(final_model)

    # Vorhersage
    predictions = []
    for _, row in test_data.iterrows():
        evidence = {f: int(row[f]) for f in features_home + features_away}
        try:
            pred = infer.query(["MatchOutcome"], evidence=evidence)
            predictions.append(np.argmax(pred.values))
        except:
            predictions.append(0)

    report = classification_report(
        test_data["MatchOutcome"], predictions, output_dict=True
    )
    result = {
        "restart": restart,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    all_results.append(result)

# Evaluation
report = classification_report(test_data["MatchOutcome"], predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report = classification_report(test_data["MatchOutcome"], predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)
