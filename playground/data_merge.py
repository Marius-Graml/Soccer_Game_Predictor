import pandas as pd

features = pd.read_csv("data/drafts/match_features_merged.csv")
labels = pd.read_csv("data/drafts/final_dataset.csv", usecols=["game_id", "result"])

# Merge und speichern
merged = features.merge(labels, on="game_id", how="inner")
merged.to_csv("data/dataset.csv", index=False)
