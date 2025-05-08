import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from clip_utils import load_clip_model, DEVICE
from flickr_dataset import FlickrImageOnly

#
# SETUP
#
#

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'openai'
SPLIT = 'test'
# loading all 31783 images from the Flickr30k dataset
SAMPLE_SIZE = 31783
BATCH_SIZE = 32
OUTPUT_PATH = 'clip_image_embeddings.parquet'

#
#
# MAIN
#
#


def main():
  # load model + transform
  model = load_clip_model(MODEL_NAME, PRETRAINED, quick_gelu=True)
  model.to(DEVICE).eval()

  # dataset from flickr_dataset.py
  dataset = FlickrImageOnly(split="train", sample_size=31783)
  loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
  )

  records = []
  print(f"Encoding in batches of {BATCH_SIZE}…")
  for batch in tqdm(loader, total=SAMPLE_SIZE // BATCH_SIZE):
    imgs = batch.to(DEVICE, non_blocking=True)
    with torch.no_grad():
      embs = model.encode_image(imgs).cpu().numpy()
    records.extend({'embedding': e} for e in embs)
    if len(records) >= SAMPLE_SIZE:
      break

  # save embeddings to parquet file
  pd.DataFrame(records).to_parquet(OUTPUT_PATH, index=False)
  print(f"Saved {len(records)} embeddings to {OUTPUT_PATH}")

#
#
#
#
#

if __name__ == '__main__':
  main()
