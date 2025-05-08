import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from caption_model import ImageCaptioningModel
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from open_clip import get_tokenizer

#
#
# DATASET
#
#

class CaptioningDataset(Dataset):
  def __init__(self, parquet_path):
    self.df = pd.read_parquet(parquet_path)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image_emb = torch.tensor(row['embedding'], dtype=torch.float32)
    cap_input = torch.tensor(row['caption_input'], dtype=torch.long)
    cap_label = torch.tensor(row['caption_label'], dtype=torch.long)
    image_path = row.get('image_path', None)
    return image_emb, cap_input, cap_label, image_path

#
#
# GREEDY DECODING
#
#

def greedy_decode(model, image_emb, tokenizer, max_len=32):
  sos_id = tokenizer.encoder['<start_of_text>']
  eos_id = tokenizer.encoder['<end_of_text>']

  tokens = [sos_id]
  for _ in range(max_len):
      input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(image_emb.device)
      logits = model(images=image_emb.unsqueeze(0), captions=input_tensor)
      next_token = logits[0, -1].argmax().item()
      tokens.append(next_token)
      if next_token == eos_id:
          break
  return tokens

#
#
# DECODE
#
#

def decode_clip_tokens(token_ids, tokenizer):
  inv_vocab = {v: k for k, v in tokenizer.encoder.items()}
  return ' '.join([inv_vocab[t] for t in token_ids if t in inv_vocab and '<' not in inv_vocab[t]])

#
#
# EVALUATION
#
#

def evaluate():
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = get_tokenizer('ViT-B-32')

  dataset = CaptioningDataset("caption_test_data.parquet")
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

  model = ImageCaptioningModel(
      decoder_vocab_size=tokenizer.vocab_size,
      decoder_max_len=30
  ).to(DEVICE)
  model.load_state_dict(torch.load("checkpoints/clip_caption_model.pth", map_location=DEVICE))
  model.eval()

  preds = {}
  gts = {}
  smoothing = SmoothingFunction()

  for idx, (image_emb, _, cap_label, image_path) in enumerate(tqdm(dataloader)):
      image_emb = image_emb.to(DEVICE)
      cap_label = cap_label.squeeze(0).tolist()
      gt_tokens = decode_clip_tokens([t for t in cap_label if t != 0], tokenizer)

      pred_ids = greedy_decode(model, image_emb.squeeze(0), tokenizer)
      pred_tokens = decode_clip_tokens([t for t in pred_ids if t != 0], tokenizer)

      preds[idx] = [pred_tokens]
      gts[idx] = [gt_tokens]

      if idx < 5:
          print(f"\nSample {idx}:")
          print(f"GT:   {gt_tokens}")
          print(f"PRED: {pred_tokens}")
          if image_path[0]:
              img = Image.open(image_path[0])
              plt.imshow(img)
              plt.axis('off')
              plt.title(f"Pred: {pred_tokens}", fontsize=10)
              plt.show()

  bleu_scores = [sentence_bleu(gts[i], preds[i][0], smoothing_function=smoothing.method1) for i in preds]
  meteor_scores = [meteor_score(gts[i], preds[i][0]) for i in preds]

  avg_bleu = np.mean(bleu_scores)
  avg_meteor = np.mean(meteor_scores)

  print(f"\nAverage BLEU score: {avg_bleu:.4f}")
  print(f"Average METEOR score: {avg_meteor:.4f}")

  cider_refs = [{"image_id": k, "captions": gts[k]} for k in gts]
  cider_hyps = [{"image_id": k, "caption": preds[k][0]} for k in preds]

  with open("refs.json", "w") as f:
      json.dump(cider_refs, f)
  with open("hyps.json", "w") as f:
      json.dump(cider_hyps, f)

  cider = Cider()
  cider_score, _ = cider.compute_score(cider_refs, cider_hyps)
  print(f"Average CIDEr score: {cider_score:.4f}")

  os.remove("refs.json")
  os.remove("hyps.json")

if __name__ == '__main__':
  evaluate()
