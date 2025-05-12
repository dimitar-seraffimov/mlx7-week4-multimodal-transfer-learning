import streamlit as st
from PIL import Image
import os
import torch
import sys
import os
# Add the parent directory to sys.path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_image_caption.caption_model import ImageCaptioningModel
from open_clip import create_model_and_transforms, get_tokenizer
from streamlit_app.db_utils import ensure_database_setup, save_caption, get_recent_captions

#
#
# SETUP
#
#

ARTIFACT_NAME = 'mlx7-dimitar-projects/mlx7-week4-multimodal/week4f_lickr30k_2025_05_08__12_46_54:latest'
CHECKPOINT_FILENAME = 'clip_caption_model_local.pth'
LOCAL_CHECKPOINT_DIR = './downloaded_artifacts'
MODEL_CHECKPOINT = os.path.join(LOCAL_CHECKPOINT_DIR, CHECKPOINT_FILENAME)
CLIP_MODEL = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 32
DB_PATH = 'captions.db'

tokenizer = get_tokenizer(CLIP_MODEL)
sos_id = tokenizer.encoder.get('<start_of_text>', 49406)
eos_id = tokenizer.encoder.get('<end_of_text>', 49407)
ensure_database_setup()

#
#
# LOAD MODELS
#
#

@st.cache_resource
def load_clip_encoder():
  model, _, preprocess = create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
  model = model.to(DEVICE)
  model.eval()
  for p in model.parameters():
      p.requires_grad = False
  return model, preprocess

@st.cache_resource
def load_decoder_model():
  vocab_size = tokenizer.vocab_size
  model = ImageCaptioningModel(decoder_vocab_size=vocab_size, decoder_max_len=MAX_LEN).to(DEVICE)
  model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
  model.eval()
  return model

#
#
# DECODE
#
#

@torch.no_grad()
def greedy_decode(model, image_embedding):
  tokens = [sos_id]
  for _ in range(MAX_LEN):
    input_ids = torch.tensor(tokens, device=DEVICE).unsqueeze(0)
    logits = model.decoder(input_ids, image_embedding.unsqueeze(0).unsqueeze(1))
    next_token = logits[0, -1].argmax().item()
    if next_token == eos_id:
        break
    tokens.append(next_token)
  return tokenizer.decode(tokens[1:]).strip()

#
#
# UI
#
#

st.title("Image Captioning Demo")

source = st.radio("Choose Image Source:", ('Upload from device', 'Take a photo'))

uploaded_image = None
if source == 'Upload from device':
  uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
elif source == 'Take a photo (mobile only)':
  uploaded_image = st.camera_input("Take a photo")

if uploaded_image:
  img = Image.open(uploaded_image).convert("RGB")
  st.image(img, caption="Input Image", use_container_width=True)

  st.write("Generating caption...")
  clip_model, preprocess = load_clip_encoder()
  decoder_model = load_decoder_model()

  image_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
  with torch.no_grad():
    image_embedding = clip_model.encode_image(image_tensor).squeeze(0)

  caption = greedy_decode(decoder_model, image_embedding)
  st.success("Caption:")
  st.markdown(f"### {caption}")

  # save image and predicted caption to DB
  os.makedirs("saved_images", exist_ok=True)
  image_path = f"saved_images/img_{len(os.listdir('saved_images'))+1}.jpg"
  # save the image and caption to the DB
  save_caption(image_path, caption)

# show history
st.sidebar.title("History")
# query the DB for the most recent captions
for img_path, cap in  get_recent_captions():
  st.sidebar.image(img_path, width=150)
  st.sidebar.caption(cap)
