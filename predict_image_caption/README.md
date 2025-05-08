## Image Captioning with CLIP Encoder and Custom Decoder

**Project Objective:**
Generate natural language descriptions for images from the Flickr30k dataset using a pretrained CLIP encoder and a Transformer decoder.

---

### Files (in order of creation):

**`s01_dataset_flickr30k.py`**
_Internal functionality:_

- load Flickr30k dataset from HuggingFace -> no download, the dataset is huge and I don't need to create embeddings on it
- use the pre-trained SentencePiece model
- extract images and corresponding 5 captions
- tokenize captions using SentencePiece
- generate `(image, caption_input, caption_label)` triples
- save preprocessed data as 'flickr30k_cache.pt' for efficient training at s03_training.py

---

**`s02_clip_encoder.py`**
_Internal functionality:_

- load pretrained CLIP model from OpenCLIP (vision + text encoder)
- encode images using `CLIPVisionTransformer`
- store visual embeddings for reuse
- optional: encode captions using `CLIPTextTransformer` for semantic comparison/analysis

---

**`decoder_transformer.py`**
_Internal functionality:_

- define Transformer decoder block:

  - masked self-attention over token inputs
  - cross-attention over image embeddings
  - feed-forward network + residuals + LayerNorm

- input: caption token sequence `[<sos>, ..., ]`
- output: logits over vocabulary

---

**`caption_model.py`**
_Internal functionality:_

- combine pretrained CLIP encoder with custom Transformer decoder
- forward pass:

  - input image → CLIP vision encoder → visual embeddings
  - input caption → decoder → predicted caption logits

- return: sequence logits for training

---

**`s03_training.py`**
_Internal functionality:_

- load preprocessed dataset triples
- initialize decoder model (CLIP encoder frozen)
- compute cross-entropy loss between predicted and true caption tokens
- log training metrics using `wandb`
- save model checkpoint after training

---

**`evaluate.py`**
_Internal functionality:_

- load trained model checkpoint
- evaluate accuracy of predicted captions against ground-truth
- log BLEU/METEOR/CIDEr scores
- visualize sample predictions for sanity checks

---

**`s04_inference.py`**
_Internal functionality:_

- load trained model
- take raw image input
- encode via frozen CLIP encoder
- generate caption via greedy decoding loop starting from `<sos>`
- visualize image + generated caption

---

**Example Output:**
![Sample output](predictions/final_output_1.jpg)
