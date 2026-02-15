import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# =========================
# CONFIG
# =========================
HF_DATASET = "dash-binayak92/tourism-purchase-prediction-processed"
MODEL_NAME = "tourism-purchase-model"

mlflow.set_experiment("Tourism Purchase Prediction")

# =========================
# 1. LOAD DATA FROM HF
# =========================
dataset = load_dataset(HF_DATASET)

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# =========================
# 2. ENCODING
# =========================
cat_cols = train_df.select_dtypes(include="object").columns

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    le_dict[col] = le

# split
X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 3. MODELS + PARAM GRID
# =========================

models = {
    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators": [100,200], "max_depth": [5,10,None]}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(),
        {"n_estimators": [100,200], "learning_rate":[0.05,0.1]}
    )
}

best_accuracy = 0
best_model = None
best_model_name = ""

# =========================
# 4. TRAIN + TRACK
# =========================
for name, (model, params) in models.items():
    with mlflow.start_run(run_name=name):

        grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # log params
        mlflow.log_params(grid.best_params_)

        # log metric
        mlflow.log_metric("accuracy", acc)

        # log model
        mlflow.sklearn.log_model(grid.best_estimator_, name)

        print(f"{name} Accuracy:", acc)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = grid.best_estimator_
            best_model_name = name

# =========================
# 5. SAVE BEST MODEL
# =========================
joblib.dump(best_model, "models/best_model.pkl")
print("Best Model:", best_model_name, best_accuracy)