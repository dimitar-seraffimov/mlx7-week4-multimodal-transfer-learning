import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from clip_utils import load_clip_model, DEVICE

#
# SETUP
#
#

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'openai'
SPLIT = 'test'
SAMPLE_SIZE = 500
BATCH_SIZE = 32
OUTPUT_PATH = 'clip_image_embeddings.parquet'

#
#
#
#
#

class FlickrIterDataset(IterableDataset):
  def __init__(self, split, preprocess, limit):
    ds = load_dataset("nlphuji/flickr30k", split=split, streaming=True)
    self.stream = iter(ds)
    self.preproc = preprocess
    self.limit = limit

  def __iter__(self):
    for i, row in enumerate(self.stream):
      if i >= self.limit:
        break
      img = row['image'].convert('RGB')
      yield self.preproc(img)  # returns a tensor [3,H,W]

#
#
# MAIN
#
#


def main():
  # load model + transform
  model, preprocess = load_clip_model(MODEL_NAME, PRETRAINED, quick_gelu=True)
  model.to(DEVICE).eval()

  # dataset + dataloader
  ds     = FlickrIterDataset(SPLIT, preprocess, SAMPLE_SIZE)
  loader = DataLoader(
    ds,
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

  # save
  pd.DataFrame(records).to_parquet(OUTPUT_PATH, index=False)
  print(f"Saved {len(records)} embeddings to {OUTPUT_PATH}")

if __name__ == '__main__':
  main()
