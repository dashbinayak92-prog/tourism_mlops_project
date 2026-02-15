from datasets import load_dataset, DatasetDict,Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
HF_DATASET = "dash-binayak92/tourism-purchase-prediction"

# =========================
# 1. LOAD DATA FROM HF
# =========================
print("Downloading dataset from HuggingFace...")
dataset = load_dataset(HF_DATASET)

df = dataset['train'].to_pandas()
print("Data Loaded:", df.shape)

# =========================
# 2. DATA CLEANING
# =========================

# Remove unnecessary column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Fill missing values
for col in df.select_dtypes(include=['int64','float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing Values After Cleaning:")
print(df.isnull().sum())

# =========================
# 3. TRAIN TEST SPLIT
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

# =========================
# 4. SAVE LOCALLY
# =========================
train_df.to_csv("artifacts/train.csv", index=False)
test_df.to_csv("artifacts/test.csv", index=False)

print("Saved locally in artifacts/")

# =========================
# 5. UPLOAD BACK TO HF
# =========================

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dataset_dict.push_to_hub(HF_DATASET + "-processed")

print("Processed dataset uploaded successfully!")