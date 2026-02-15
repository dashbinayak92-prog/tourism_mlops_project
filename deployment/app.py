import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# download model from HF hub
model_path = hf_hub_download(
    repo_id="dash-binayak92/tourism-purchase-model",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

def predict(
    Age, CityTier, DurationOfPitch, NumberOfPersonVisiting,
    NumberOfFollowups, NumberOfTrips, MonthlyIncome
):
    df = pd.DataFrame([[
        Age, CityTier, DurationOfPitch, NumberOfPersonVisiting,
        NumberOfFollowups, NumberOfTrips, MonthlyIncome
    ]], columns=[
        "Age","CityTier","DurationOfPitch","NumberOfPersonVisiting",
        "NumberOfFollowups","NumberOfTrips","MonthlyIncome"
    ])

    prediction = model.predict(df)[0]

    return "Customer WILL Purchase Package" if prediction==1 else "Customer NOT Interested"

interface = gr.Interface(
    fn=predict,
    inputs=[gr.Number() for _ in range(7)],
    outputs="text",
    title="Tourism Package Purchase Predictor"
)

interface.launch(server_name="0.0.0.0", server_port=7860)