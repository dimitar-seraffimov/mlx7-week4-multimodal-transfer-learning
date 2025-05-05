import torch
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
from caption_model import ImageCaptioningModel
from open_clip import get_tokenizer
import requests
from io import BytesIO
import os


#
#
# SETUP
#
#

MODEL_CHECKPOINT = 'checkpoints/clip_caption_model.pth'
CLIP_MODEL = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 30

#
#
# LOAD COMPONENTS
#
#

def load_clip_encoder(model_name, pretrained, device):
  model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
  model = model.to(device)
  model.eval()
  for p in model.parameters():
      p.requires_grad = False
  return model, preprocess

#
#
# GREEDY DECODING
#
#

def greedy_decode(model, image_embedding, tokenizer, max_len=30):
  sos_id = tokenizer.encoder['<|startoftext|>']
  eos_id = tokenizer.encoder['<|endoftext|>']

  tokens = [sos_id]
  for _ in range(max_len):
      input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(image_embedding.device)
      logits = model(images=image_embedding.unsqueeze(0), captions=input_tensor)  # [1, T, V]
      next_token = logits[0, -1].argmax().item()
      tokens.append(next_token)
      if next_token == eos_id:
          break
  return tokens

#
#
# INFERENCE FUNCTION
#
#


def decode_clip_tokens(token_ids, tokenizer):
  inv_vocab = {v: k for k, v in tokenizer.encoder.items()}
  return ' '.join([inv_vocab[t] for t in token_ids if t in inv_vocab and '<' not in inv_vocab[t]])

def generate_caption(image_path):
  tokenizer = get_tokenizer(CLIP_MODEL)

  # load CLIP encoder
  clip_model, preprocess = load_clip_encoder(CLIP_MODEL, CLIP_PRETRAINED, DEVICE)

  # load image and encode
  image = Image.open(image_path).convert("RGB")
  image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

  with torch.no_grad():
      image_embedding = clip_model.encode_image(image_tensor).squeeze(0)

  # Load decoder model
  vocab_size = tokenizer.vocab_size
  model = ImageCaptioningModel(
    decoder_vocab_size=vocab_size,
    decoder_max_len=MAX_LEN
  ).to(DEVICE)
  model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
  model.eval()

  # decode caption
  token_ids = greedy_decode(model, image_embedding, tokenizer, max_len=MAX_LEN)
  caption = decode_clip_tokens(token_ids[1:], tokenizer)  # skip <sos>

  # show image + caption
  plt.imshow(image)
  plt.axis("off")
  plt.title(f"Predicted caption:\n{caption}", fontsize=12)
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