# run_streamlit.py (adjusted version for your project)
import os
import subprocess
import sys

def check_dependencies():
  """Ensure all dependencies are installed."""
  try:
    import streamlit
    import torch
    import PIL
    import sqlite3
  except ImportError:
    print("Missing dependencies. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def ensure_checkpoint():
  """Ensure the model checkpoint is available."""
  checkpoint_dir = 'downloaded_artifacts'
  checkpoint_file = 'clip_caption_model_local.pth'
  checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
  if not os.path.exists(checkpoint_path):
    print("Checkpoint not found. Make sure to provide the model.")
  else:
    os.environ['MODEL_CHECKPOINT'] = checkpoint_path


def run_streamlit():
  """Run the Streamlit app."""
  port = int(os.environ.get("PORT", 8501))
  os.environ["STREAMLIT_SERVER_PORT"] = str(port)
  os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
  os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
  subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)


if __name__ == "__main__":
  check_dependencies()
  ensure_checkpoint()
  run_streamlit()
