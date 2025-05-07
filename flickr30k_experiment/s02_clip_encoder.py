import torch
from PIL import Image
from datasets import load_dataset
import open_clip
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from clip_utils import load_clip_model, DEVICE

#
# SETUP
#
#

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'openai'
SPLIT = 'test'
SAMPLE_SIZE = 5000
OUTPUT_PATH = 'clip_image_embeddings.parquet'

#
#
# MAIN
#
#

def main():
  # load the nlphuji/flickr30k split
  dataset = load_dataset("nlphuji/flickr30k", split=SPLIT, streaming=True)
  stream = iter(dataset)
  model, preprocess = load_clip_model(MODEL_NAME, PRETRAINED, quick_gelu=True)
  print(f"Model loaded on {DEVICE}")
  records = []
  print(f"Streaming and encoding up to {SAMPLE_SIZE} images...")

  count = 0

  # encode each image
  for row in tqdm(stream, total=SAMPLE_SIZE, desc="Encoding images"):
    img = row['image'].convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
      emb = model.encode_image(img_tensor).squeeze(0).cpu().numpy()
    records.append({'embedding': emb})

    count += 1
    if count >= SAMPLE_SIZE:
      break

  # save embeddings
  df = pd.DataFrame(records)
  df.to_parquet(OUTPUT_PATH, index=False)
  print(f"Saved {len(df)} embeddings to {OUTPUT_PATH}")


if __name__ == '__main__':
  main()
