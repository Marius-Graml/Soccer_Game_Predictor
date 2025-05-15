# Re-run soft-EM adjusted code after environment reset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

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

kbins_home = KBinsDiscretizer(n_bins=__BINS, encode="ordinal", strategy="uniform")
kbins_away = KBinsDiscretizer(n_bins=__BINS, encode="ordinal", strategy="uniform")
data[features_home] = kbins_home.fit_transform(data[features_home]).astype(int)
data[features_away] = kbins_away.fit_transform(data[features_away]).astype(int)

# Initialize soft latent variable columns with uniform priors
for i in range(__BINS):
    data[f"HomeStrength_{i}"] = 1 / __BINS
    data[f"AwayStrength_{i}"] = 1 / __BINS

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

# KMeans based initialization of latent variables
kmeans_home = KMeans(n_clusters=__BINS, random_state=42)
kmeans_away = KMeans(n_clusters=__BINS, random_state=42)

# Determine cluster labels (only apply on train data)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train KMeans on home- and away- features in train set
train_data["HomeStrength"] = kmeans_home.fit_predict(train_data[features_home])
train_data["AwayStrength"] = kmeans_away.fit_predict(train_data[features_away])

# Initialize latent distributions ("soft priors") based on clusters
for i in range(__BINS):
    train_data[f"HomeStrength_{i}"] = (train_data["HomeStrength"] == i).astype(float)
    train_data[f"AwayStrength_{i}"] = (train_data["AwayStrength"] == i).astype(float)

# Soft-EM
n_iterations = 5
for iteration in range(n_iterations):
    print("Iteration " + str(iteration + 1))
    # Convert expected values into hard assignments for fitting (approximation)
    train_data["HomeStrength"] = train_data[
        [f"HomeStrength_{i}" for i in range(__BINS)]
    ].values.argmax(axis=1)
    train_data["AwayStrength"] = train_data[
        [f"AwayStrength_{i}" for i in range(__BINS)]
    ].values.argmax(axis=1)

    model = BayesianNetwork(structure)
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
    infer = VariableElimination(model)

    # Update expected probabilities (soft assignment)
    for i, row in train_data.iterrows():
        evidence_home = {f: int(row[f]) for f in features_home}
        evidence_away = {f: int(row[f]) for f in features_away}
        try:
            home_post = infer.query(["HomeStrength"], evidence=evidence_home)
            away_post = infer.query(["AwayStrength"], evidence=evidence_away)
            for s in range(__BINS):
                train_data.at[i, f"HomeStrength_{s}"] = home_post.values[s]
                train_data.at[i, f"AwayStrength_{s}"] = away_post.values[s]
        except:
            continue

# Final model and test prediction
train_data["HomeStrength"] = train_data[
    [f"HomeStrength_{i}" for i in range(__BINS)]
].values.argmax(axis=1)
train_data["AwayStrength"] = train_data[
    [f"AwayStrength_{i}" for i in range(__BINS)]
].values.argmax(axis=1)

final_model = BayesianNetwork(structure)
final_model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
infer = VariableElimination(final_model)

predictions = []
for _, row in test_data.iterrows():
    evidence = {f: int(row[f]) for f in features_home + features_away}
    try:
        pred = infer.query(["MatchOutcome"], evidence=evidence)
        predictions.append(np.argmax(pred.values))
    except:
        predictions.append(0)

# Evaluation
report = classification_report(test_data["MatchOutcome"], predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report = classification_report(test_data["MatchOutcome"], predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)
