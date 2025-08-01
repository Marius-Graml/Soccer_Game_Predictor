import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--em-type", choices=["hard", "soft"], default="hard")
parser.add_argument("--balanced", type=bool, default=False)
parser.add_argument(
    "--init", choices=["random", "uniform", "kmeans", "multi"], default="random"
)
parser.add_argument("--bins", type=int, default=3)
parser.add_argument("--iterations", type=int, default=5)
parser.add_argument("--restarts", type=int, default=3)
parser.add_argument(
    "--dataset", choices=["small", "full"], default="small", help="Dataset to use"
)
args = parser.parse_args()

# Load data based on dataset argument
if args.dataset == "small":
    data = pd.read_csv("data/small_dataset.csv")
else:
    data = pd.read_csv("data/dataset.csv")

# Prepare Label
label_map = {"Home Win": 0, "Draw": 1, "Away Win": 2}
data = data[data["result"].isin(label_map)]
data["MatchOutcome"] = data["result"].map(label_map)

# Balance Data
if args.balanced:
    min_class_size = data["MatchOutcome"].value_counts().min()
    data = (
        data.groupby("MatchOutcome")
        .sample(n=min_class_size, random_state=42)
        .reset_index(drop=True)
    )

# Automatically select all home/away numeric features (exclude ID and target)
excluded = ["game_id", "result", "MatchOutcome"]
features_home = [
    col
    for col in data.columns
    if "home" in col
    and col not in excluded
    and data[col].dtype in [np.float64, np.int64]
]
features_away = [
    col
    for col in data.columns
    if "away" in col
    and col not in excluded
    and data[col].dtype in [np.float64, np.int64]
]

# Discretize
kbins_home = KBinsDiscretizer(n_bins=args.bins, encode="ordinal", strategy="uniform")
kbins_away = KBinsDiscretizer(n_bins=args.bins, encode="ordinal", strategy="uniform")
data[features_home] = kbins_home.fit_transform(data[features_home]).astype(int)
data[features_away] = kbins_away.fit_transform(data[features_away]).astype(int)


def build_structure():
    return (
        [(feat, "HomeStrength") for feat in features_home]
        + [(feat, "AwayStrength") for feat in features_away]
        + [("HomeStrength", "MatchOutcome"), ("AwayStrength", "MatchOutcome")]
    )


def initialize_soft(data):
    for i in range(args.bins):
        data[f"HomeStrength_{i}"] = 1 / args.bins
        data[f"AwayStrength_{i}"] = 1 / args.bins
    return data


def em_soft(train_data, structure):
    for iteration in range(args.iterations):
        print(f"Soft-EM Iteration {iteration+1}")
        train_data["HomeStrength"] = train_data[
            [f"HomeStrength_{i}" for i in range(args.bins)]
        ].values.argmax(axis=1)
        train_data["AwayStrength"] = train_data[
            [f"AwayStrength_{i}" for i in range(args.bins)]
        ].values.argmax(axis=1)
        model = BayesianNetwork(structure)
        model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
        infer = VariableElimination(model)

        for i, row in train_data.iterrows():
            try:
                home_post = infer.query(
                    ["HomeStrength"], evidence={f: int(row[f]) for f in features_home}
                )
                away_post = infer.query(
                    ["AwayStrength"], evidence={f: int(row[f]) for f in features_away}
                )
                for s in range(args.bins):
                    train_data.at[i, f"HomeStrength_{s}"] = home_post.values[s]
                    train_data.at[i, f"AwayStrength_{s}"] = away_post.values[s]
            except:
                continue
    return train_data


def em_hard(train_data, structure):
    for iteration in range(args.iterations):
        print(f"Hard-EM Iteration {iteration+1}")
        model = BayesianNetwork(structure)
        model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
        infer = VariableElimination(model)

        for i, row in train_data.iterrows():
            try:
                home_post = infer.query(
                    ["HomeStrength"], evidence={f: int(row[f]) for f in features_home}
                )
                away_post = infer.query(
                    ["AwayStrength"], evidence={f: int(row[f]) for f in features_away}
                )
                train_data.at[i, "HomeStrength"] = np.argmax(home_post.values)
                train_data.at[i, "AwayStrength"] = np.argmax(away_post.values)
            except:
                continue
    return train_data


results = []
restarts = args.restarts if args.init == "multi" else 1
for restart in range(restarts):
    train_data, test_data = train_test_split(
        data.copy(), test_size=0.2, random_state=42 + restart
    )
    structure = build_structure()

    if args.em_type == "soft":
        train_data = initialize_soft(train_data)
        if args.init == "kmeans":
            kmeans_home = KMeans(n_clusters=args.bins, random_state=42)
            kmeans_away = KMeans(n_clusters=args.bins, random_state=42)
            train_data["HomeStrength"] = kmeans_home.fit_predict(
                train_data[features_home]
            )
            train_data["AwayStrength"] = kmeans_away.fit_predict(
                train_data[features_away]
            )
            mean_strengths = (
                train_data.groupby("HomeStrength")[features_home[0]]
                .mean()
                .sort_values()
            )
            label_map = {old: new for new, old in enumerate(mean_strengths.index)}
            train_data["HomeStrength"] = train_data["HomeStrength"].map(label_map)

            # pca = PCA(n_components=2)
            # X_pca = pca.fit_transform(train_data[features_home])
            # plt.scatter(
            #     X_pca[:, 0],
            #     X_pca[:, 1],
            #     c=train_data["HomeStrength"],
            #     cmap="viridis",
            #     alpha=0.6,
            # )
            # plt.title("HomeStrength Clusters via PCA")
            # plt.xlabel("PCA1")
            # plt.ylabel("PCA2")
            # plt.colorbar(label="HomeStrength Cluster")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

        if args.init in ["kmeans", "random"]:
            if args.init == "random":
                train_data["HomeStrength"] = np.random.randint(
                    0, args.bins, len(train_data)
                )
                train_data["AwayStrength"] = np.random.randint(
                    0, args.bins, len(train_data)
                )
            for i in range(args.bins):
                train_data[f"HomeStrength_{i}"] = (
                    train_data["HomeStrength"] == i
                ).astype(float)
                train_data[f"AwayStrength_{i}"] = (
                    train_data["AwayStrength"] == i
                ).astype(float)
        train_data = em_soft(train_data, structure)
    else:
        train_data["HomeStrength"] = np.random.randint(0, args.bins, len(train_data))
        train_data["AwayStrength"] = np.random.randint(0, args.bins, len(train_data))
        train_data = em_hard(train_data, structure)

    final_model = BayesianNetwork(structure)
    final_model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
    infer = VariableElimination(final_model)

    predictions = []
    for _, row in test_data.iterrows():
        try:
            pred = infer.query(
                ["MatchOutcome"],
                evidence={f: int(row[f]) for f in features_home + features_away},
            )
            predictions.append(np.argmax(pred.values))
        except:
            predictions.append(0)

    report = classification_report(
        test_data["MatchOutcome"], predictions, output_dict=True
    )
    print("\n=== Classification Report ===")
    print(pd.DataFrame(report).transpose())

    print("\n=== CPT of MatchOutcome ===")
    print(final_model.get_cpds("MatchOutcome"))

    print("\n=== Average home team features per HomeStrength cluster ===")
    print(train_data.groupby("HomeStrength")[features_home].mean())

    example_evidence = {
        "home_avg_market_value": 2,
        "home_avg_age": 0,
        "home_nationalities": 2,
        "home_total_minutes": 2,
    }
    print("\n=== Posterior for example home team ===")
    print(infer.query(["HomeStrength"], evidence=example_evidence))

    print("\n=== Scenario: Strong home vs weak away ===")
    print(
        infer.query(["MatchOutcome"], evidence={"HomeStrength": 2, "AwayStrength": 0})
    )
