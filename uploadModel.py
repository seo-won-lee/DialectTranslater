from huggingface_hub import HfApi

# ✅ Hugging Face API 사용
api = HfApi()

# ✅ Hugging Face에 모델 업로드
repo_id = "thisischloe/dialectTranslater"
api.create_repo(repo_id=repo_id, exist_ok=True)

# ✅ 모델 파일을 업로드합니다.
api.upload_folder(
    folder_path="./dialectTranslater",  # 업로드할 폴더 경로
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial model upload"
)

print(f"✅ Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_id}")
