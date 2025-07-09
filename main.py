from scripts.feature_extraction import extract_features
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
import os
import pandas as pd
from tqdm import tqdm

csv_path = "data/train.csv"
image_dir = "data/images"
df = pd.read_csv(csv_path)

X, y = [], []

print("Extracting features...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    label = row['label']
    path = os.path.join(image_dir, row['image_id'] + ".jpg")
    features = extract_features(path)
    X.append(features)
    y.append(label)

feature_df = pd.DataFrame(X)
feature_df['label'] = y
feature_df.to_csv("features/features.csv", index=False)

train_model(feature_df.drop(columns=['label']), feature_df['label'], "models/rf_model.pkl")
evaluate_model("models/rf_model.pkl", feature_df.drop(columns=['label']), feature_df['label'])
