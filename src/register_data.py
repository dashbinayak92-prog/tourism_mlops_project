from datasets import load_dataset, Dataset
import pandas as pd

# Load local data
df = pd.read_csv("data/tourism.csv")

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Push to hub
dataset.push_to_hub("dash-binayak92/tourism-purchase-prediction")