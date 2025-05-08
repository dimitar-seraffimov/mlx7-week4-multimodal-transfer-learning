import torch
import torch.nn as nn
from decoder_transformer import TransformerDecoder
from clip_utils import load_clip_model


class ImageCaptioningModel(nn.Module):
  def __init__(self, 
        decoder_vocab_size: int,
        decoder_max_len: int,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 8,
        decoder_ff_dim: int = 2048,
        decoder_num_layers: int = 6,
        clip_model_name: str = 'ViT-B-32',
        clip_pretrained: str = 'openai',
        quick_gelu: bool = True
    ):
    super().__init__()

    # CLIP encoder (frozen)
    self.clip_model, self.clip_preprocess = load_clip_model(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        quick_gelu=quick_gelu
    )

    # Transformer decoder
    self.decoder = TransformerDecoder(
        vocab_size=decoder_vocab_size,
        max_len=decoder_max_len,
        embed_dim=decoder_embed_dim,
        num_heads=decoder_num_heads,
        ff_dim=decoder_ff_dim,
        num_layers=decoder_num_layers
    )

  def forward(self, images, captions, caption_pad_mask=None):
    """
    Args:
        images: Tensor of shape [B, 3, H, W]
        captions: Tensor of shape [B, T] (input tokens)
        caption_pad_mask: Optional [B, T] mask for padded tokens
    Returns:
        logits: Tensor of shape [B, T, vocab_size]
    """
    with torch.no_grad():
        image_embeddings = self.clip_model.encode_image(images)  # [B, D]
    
    # add sequence dim to match decoder format: [B, 1, D]
    memory = image_embeddings.unsqueeze(1)

    logits = self.decoder(tgt_input=captions, memory=memory, tgt_pad_mask=caption_pad_mask)
    return logits
