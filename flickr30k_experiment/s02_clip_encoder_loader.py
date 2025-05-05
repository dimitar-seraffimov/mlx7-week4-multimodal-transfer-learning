import torch
from PIL import Image
from datasets import load_dataset
import open_clip
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

#
#
# SETUP
#
#

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'openai'
SPLIT = 'test'
OUTPUT_PATH = 'clip_image_embeddings.parquet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#
#
# LOAD CLIP MODEL
#
#

def load_clip_model():
  # Load pretrained CLIP and preprocess transforms
  model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
  model = model.to(DEVICE)
  model.eval()
  for p in model.parameters():
      p.requires_grad = False
  return model, preprocess

#
#
# MAIN
#
#

def main():
  # load the nlphuji/flickr30k split
  ds = load_dataset('nlphuji/flickr30k', split=SPLIT)

  model, preprocess = load_clip_model()
  records = []

  # encode each image
  for row in tqdm(ds, desc=f"Encoding {SPLIT} images", total=len(ds)):
      img_field = row['image']
      # handle different image field types
      if isinstance(img_field, dict) and 'path' in img_field:
          img = Image.open(img_field['path']).convert('RGB')
          img_path = img_field['path']
      elif isinstance(img_field, Image.Image):
          img = img_field
          img_path = None
      else:
          # fallback: assume Hugging Face returns bytes
          img = Image.open(img_field['bytes']).convert('RGB')
          img_path = None

      # preprocess & encode
      img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
      with torch.no_grad():
          emb = model.encode_image(img_tensor).squeeze(0).cpu().numpy()

      records.append({'image_path': img_path, 'embedding': emb})

  # save embeddings
  df = pd.DataFrame(records)
  df.to_parquet(OUTPUT_PATH, index=False)
  print(f"Saved {len(df)} embeddings to {OUTPUT_PATH}")


if __name__ == '__main__':
  main()
