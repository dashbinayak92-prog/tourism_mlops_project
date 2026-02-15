from huggingface_hub import HfApi, upload_folder

api = HfApi()

repo_id = "dash-binayak92/tourism-app"

api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

upload_folder(
    folder_path="deployment",
    repo_id=repo_id,
    repo_type="space"
)

print("App deployed!")