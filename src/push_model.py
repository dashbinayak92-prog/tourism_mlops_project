from huggingface_hub import HfApi, upload_file

api = HfApi()

repo_id = "dash-binayak92/tourism-purchase-model"

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

upload_file(
    path_or_fileobj="models/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=repo_id,
    repo_type="model"
)

print("Model uploaded to HF Hub!")