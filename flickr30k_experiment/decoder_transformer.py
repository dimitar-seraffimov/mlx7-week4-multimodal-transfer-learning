import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
# TRANSFORMER DECODER BLOCK
#
#

class TransformerDecoderBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
      super().__init__()
      self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
      self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

      self.ffn = nn.Sequential(
          nn.Linear(embed_dim, ff_dim),
          nn.ReLU(),
          nn.Linear(ff_dim, embed_dim)
      )

      self.norm1 = nn.LayerNorm(embed_dim)
      self.norm2 = nn.LayerNorm(embed_dim)
      self.norm3 = nn.LayerNorm(embed_dim)

      self.dropout = nn.Dropout(dropout)

  def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
      # masked self-attention
      attn1, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
      tgt = self.norm1(tgt + self.dropout(attn1))

      # cross-attention over encoder memory (image embeddings)
      attn2, _ = self.cross_attn(tgt, memory, memory)
      tgt = self.norm2(tgt + self.dropout(attn2))

      # feed-forward + residual
      ffn_out = self.ffn(tgt)
      tgt = self.norm3(tgt + self.dropout(ffn_out))
      return tgt

#
#
# TRANSFORMER DECODER
#
#

class TransformerDecoder(nn.Module):
  def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
      super().__init__()
      self.token_embedding = nn.Embedding(vocab_size, embed_dim)
      self.position_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

      self.decoder_blocks = nn.ModuleList([
          TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
          for _ in range(num_layers)
      ])

      self.fc_out = nn.Linear(embed_dim, vocab_size)

  def generate_square_subsequent_mask(self, sz):
      return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

  def forward(self, tgt_input, memory, tgt_pad_mask=None):
      B, T = tgt_input.size()
      device = tgt_input.device

      token_emb = self.token_embedding(tgt_input)
      pos_emb = self.position_embedding[:, :T, :]
      x = token_emb + pos_emb

      tgt_mask = self.generate_square_subsequent_mask(T).to(device)

      for block in self.decoder_blocks:
          x = block(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)

      logits = self.fc_out(x)
      return logits