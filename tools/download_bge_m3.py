from huggingface_hub import snapshot_download

patterns = []
model_dir = snapshot_download('BAAI/bge-m3')

print(model_dir)
