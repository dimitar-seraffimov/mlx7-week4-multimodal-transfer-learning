import torch
from PIL import Image
import matplotlib.pyplot as plt
from caption_model import ImageCaptioningModel
from open_clip import create_model_and_transforms, get_tokenizer
#
#
# SETUP
#
#

MODEL_CHECKPOINT = 'checkpoints/clip_caption_model_local.pth'
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

def greedy_decode(model, image_embedding, tokenizer, max_len=30):
  sos_id = tokenizer(["<|startoftext|>"])[0][0]
  eos_id = tokenizer(["<|endoftext|>"])[0][0]
  tokens = [sos_id]

  print(f"Greedy decoding with max len {max_len}...")
  for _ in range(max_len):
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    logits = model.decoder(input_ids, image_emb.unsqueeze(0).unsqueeze(1))
    next_token = logits[0, -1].argmax().item()
    if next_token == eos:
        break
    tokens.append(next_token)
  return tokens[1:] # remove <sos>

#
#
# INFERENCE FUNCTION
#
#

def generate_caption(image_path, save_output=True):
  tokenizer = get_tokenizer(CLIP_MODEL)

  # load CLIP encoder
  clip_model, preprocess = load_clip_encoder(CLIP_MODEL, CLIP_PRETRAINED, DEVICE)

  # load image and encode
  image = Image.open(image_path).convert("RGB")
  image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
  print(f"Image loaded and encoded on {DEVICE}")

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

  tokens = greedy_decode(model, image_embedding, tokenizer, max_len=MAX_LEN)
  caption = tokenizer.decode(tokens).strip()

  # show image + caption
  plt.imshow(image)
  plt.axis("off")
  plt.title(f"Predicted caption:\n{caption}", fontsize=12)
  if save_output:
      output_path = "final_output.jpg"
      plt.savefig(output_path, bbox_inches="tight")
      print(f"Saved output to {output_path}")

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