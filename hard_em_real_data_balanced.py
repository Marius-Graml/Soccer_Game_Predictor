import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report

# Load real data
data = pd.read_csv("final_dataset.csv")

# Convert target variable to numerical data
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

# Discretize continuous features
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
__BINS = 5

kbins_home = KBinsDiscretizer(n_bins=__BINS, encode="ordinal", strategy="uniform")
kbins_away = KBinsDiscretizer(n_bins=__BINS, encode="ordinal", strategy="uniform")
data[features_home] = kbins_home.fit_transform(data[features_home]).astype(int)
data[features_away] = kbins_away.fit_transform(data[features_away]).astype(int)

# Initial latent variable estimation: random (Initialization for EM algorithm)
n_samples = len(data)
print("\nNumber of samples: " + str(n_samples))
data["HomeStrength"] = np.random.randint(0, __BINS, n_samples)
data["AwayStrength"] = np.random.randint(0, __BINS, n_samples)

# Define structure of graphical model / bayesian network
# features -> team strength
# team strength -> results
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

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# EM iterations
n_iterations = 5
for iteration in range(n_iterations):
    print("Iteration " + str(iteration + 1))
    model = BayesianNetwork(structure)
    # M-step (starts with random initialization above, determine all CPTs)
    print("M-Step")
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
    # BDeu: virtual counts for all combinations of team strength -> features (= smoothing)
    # avoids probabilities of zero in CPT

    # creating an inference object for querying conditional probabilities from the joint distribution encoded by the graphical model
    infer = VariableElimination(model)

    # E-Step: update HomeStrength and AwayStrength using MAP estimates
    print("E-Step")
    for i, row in train_data.iterrows():
        evidence = {
            f: int(row[f]) for f in features_home + features_away
        }  # build dictionary from current row
        try:
            home_post = infer.query(
                ["HomeStrength"], evidence={k: evidence[k] for k in features_home}
            )
            away_post = infer.query(
                ["AwayStrength"], evidence={k: evidence[k] for k in features_away}
            )
            train_data.at[i, "HomeStrength"] = np.argmax(home_post.values)
            train_data.at[i, "AwayStrength"] = np.argmax(away_post.values)
        except:
            continue  # skip cases where inference fails due to insufficient CPTs

# Final model
final_model = BayesianNetwork(structure)
final_model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
infer = VariableElimination(final_model)

# Predict on test set
predictions = []
for _, row in test_data.iterrows():
    evidence = {f: int(row[f]) for f in features_home + features_away}
    try:
        pred = infer.query(["MatchOutcome"], evidence=evidence)
        predictions.append(np.argmax(pred.values))
    except:
        predictions.append(0)  # fallback if inference fails

# Evaluation
report = classification_report(test_data["MatchOutcome"], predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report = classification_report(test_data["MatchOutcome"], predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)
