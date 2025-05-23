import torch
from PIL import Image
import matplotlib.pyplot as plt
from caption_model import ImageCaptioningModel
from open_clip import create_model_and_transforms, get_tokenizer
import wandb
import os

#
#
# SETUP
#
#

# model artifact info
# my best trained model from the saved wandb artifact model
ARTIFACT_NAME = 'mlx7-dimitar-projects/mlx7-week4-multimodal/week4f_lickr30k_2025_05_08__12_46_54:latest'
CHECKPOINT_FILENAME = 'clip_caption_model_local.pth'
LOCAL_CHECKPOINT_DIR = './downloaded_artifacts'
MODEL_CHECKPOINT = os.path.join(LOCAL_CHECKPOINT_DIR, CHECKPOINT_FILENAME)

# download my best trained model from the saved wandb artifact model
# this model is trained on 20k samples from the Flickr30k dataset, 8 epochs

if not os.path.exists(MODEL_CHECKPOINT):
  print("Model checkpoint not found locally. Downloading from Weights & Biases...")
  wandb.login()
  run = wandb.init(project="mlx7-week4-multimodal", job_type="inference", reinit=True)
  artifact = run.use_artifact(ARTIFACT_NAME, type='model')
  artifact_dir = artifact.download(root=LOCAL_CHECKPOINT_DIR)
  MODEL_CHECKPOINT = os.path.join(artifact_dir, CHECKPOINT_FILENAME)
else:
  print("Model checkpoint found locally. Using cached version.")

CLIP_MODEL = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 32
PAD_ID = 0


# initialize tokenizer and special IDs
tokenizer = get_tokenizer(CLIP_MODEL)
sos_id = tokenizer.encoder.get('<start_of_text>', 49406)
eos_id = tokenizer.encoder.get('<end_of_text>', 49407)

#
#
# LOAD COMPONENTS
#
#

def load_clip_encoder(model_name, pretrained, device):
  model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
  model = model.to(device)
  print(f"CLIP model loaded on {device}")
  model.eval()
  for p in model.parameters():
      p.requires_grad = False
  return model, preprocess

#
#
# GREEDY DECODING
#
#

def greedy_decode(model, image_embedding, max_len=MAX_LEN):
  tokens = [sos_id]
  print(f"Greedy decoding with max len {max_len}...")
  for _ in range(max_len):
    input_ids = torch.tensor(tokens, device=DEVICE).unsqueeze(0)
    logits = model.decoder(
      input_ids,
      image_embedding.unsqueeze(0).unsqueeze(1)
    )
    next_token = logits[0, -1].argmax().item()
    if next_token == eos_id:
      break
    tokens.append(next_token)

  return tokens[1:] # drop <sos>

#
#
# INFERENCE FUNCTION
#
#

def generate_caption(image_path, save_output=True):
  clip_model, preprocess = load_clip_encoder(
    CLIP_MODEL, CLIP_PRETRAINED, DEVICE
  )

  # load and preprocess image
  image = Image.open(image_path).convert("RGB")
  image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
  print(f"Image loaded and encoded on {DEVICE}")

  # encode image
  with torch.no_grad():
    image_embedding = clip_model.encode_image(image_tensor).squeeze(0)

  # load decoder model
  vocab_size = tokenizer.vocab_size
  model = ImageCaptioningModel(
    decoder_vocab_size=vocab_size,
    decoder_max_len=MAX_LEN
  ).to(DEVICE)
  model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
  model.eval()

  # generate tokens and decode to text
  token_ids = greedy_decode(model, image_embedding)
  caption = tokenizer.decode(token_ids).strip()

  # display
  plt.imshow(image)
  plt.axis("off")
  plt.title(f"Predicted caption:\n{caption}", fontsize=12)
  if save_output:
    out = "final_output.jpg"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved output to {out}")
  plt.show()

  return caption
#
#
# MAIN
#
#

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--image', type=str, required=True, help='Path to input image')
  args = parser.parse_args()

  caption = generate_caption(args.image)
  print(f"\nGenerated Caption:\n{caption}")