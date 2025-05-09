import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

ARTIFACT_NAME = 'mlx7-dimitar-projects/mlx7-week4-multimodal/week4f_lickr30k_2025_05_08__12_46_54:latest'
CHECKPOINT_FILENAME = 'clip_caption_model_local.pth'
LOCAL_CHECKPOINT_DIR = 'downloaded_artifacts'

def check_dependencies():
  """Ensure all dependencies are installed."""
  try:
    import streamlit
    import torch
    import PIL
    import sqlite3
    import wandb
  except ImportError:
    print("Missing dependencies. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def ensure_checkpoint():
  """Ensure the model checkpoint is downloaded from wandb or available locally."""
  # create the checkpoint directory if it doesn't exist
  os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)
  checkpoint_path = os.path.join(LOCAL_CHECKPOINT_DIR, CHECKPOINT_FILENAME)
  if not os.path.exists(checkpoint_path):
    print("Checkpoint not found locally. Downloading from wandb...")
    import wandb
    wandb.login()
    run = wandb.init(project="mlx7-week4-multimodal", job_type="inference", reinit=True)
    artifact = run.use_artifact(ARTIFACT_NAME, type='model')
    artifact_dir = artifact.download(root=LOCAL_CHECKPOINT_DIR)
    checkpoint_path = os.path.join(artifact_dir, CHECKPOINT_FILENAME)
    print(f"Downloaded model checkpoint to: {checkpoint_path}")
  else:
    print("Model checkpoint found locally. Using cached version.")

  os.environ['MODEL_CHECKPOINT'] = checkpoint_path

def run_streamlit():
  """Run the Streamlit app."""
  port = int(os.environ.get("PORT", 8080))
  os.environ["STREAMLIT_SERVER_PORT"] = str(port)
  os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
  os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
  subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)


if __name__ == "__main__":
  check_dependencies()
  ensure_checkpoint()
  run_streamlit()
