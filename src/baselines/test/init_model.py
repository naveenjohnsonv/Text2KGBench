from llama_cpp import Llama
import os
import requests
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_model():
    models_dir = "/data/johnsonv/models"
    model_name = "qwen2.5-32b-instruct-q4_k_m.gguf"
    model_path = os.path.join(models_dir, model_name)
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path
    
    print("Downloading model... This may take a while.")
    try:
        # Download from Hugging Face
        model_path = hf_hub_download(
            repo_id="TheRains/Qwen2.5-32B-Instruct-Q4_K_M-GGUF",
            filename=model_name,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None

def initialize_model():
    model_path = download_model()
    if not model_path:
        return None
    
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,        # Maximum layers for RTX 3090
            n_ctx=2048,             # Keep default context size
            n_threads=24,           # Match your CPU core count
            offload_kqv=True,       # Beneficial for large models
            use_mlock=True
        )
        print("Model initialized successfully")
        return llm
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None

if __name__ == "__main__":
    model = initialize_model()
    if model:
        print("Model loaded and ready for use")
