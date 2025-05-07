import os
import torch
torch.manual_seed(42)
from torch.utils.data import DataLoader
import wandb
from caption_model import ImageCaptioningModel
from s01_dataset_flickr30k import Flickr30kCaptionDataset
from open_clip import get_tokenizer
from tqdm import tqdm
import datetime
from clip_utils import DEVICE

#
#
# SETUP
#
#

BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 30
LEARNING_RATE = 3e-4
SPLIT = 'test'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# initialize tokenizer and vocab size   
tokenizer = get_tokenizer('ViT-B-32')
vocab_size = tokenizer.vocab_size
pad_id = 0

# prepare dataset and dataloader
train_ds = Flickr30kCaptionDataset(split=SPLIT,
    max_caption_len=MAX_LEN,
    image_size=224
)
train_loader = DataLoader(train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# initialize model
model = ImageCaptioningModel(
    decoder_vocab_size=vocab_size,
    decoder_max_len=MAX_LEN,
).to(DEVICE)

print(f"Model initialized on {DEVICE} with vocab size {vocab_size} and max len {MAX_LEN}")

# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

# initialize W&B
wandb.init(
    project='mlx7-week4-multimodal',
    name=f'week4f_lickr30k_{timestamp}',
    config={
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'max_len': MAX_LEN
    }
)

wandb.watch(model, log='all')

# training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for images, caption_in, caption_label in progress_bar:
        images = images.to(DEVICE)           # [B,3,H,W]
        caption_in = caption_in.to(DEVICE)           # [B,T]
        caption_label = caption_label.to(DEVICE)         # [B,T]

        optimizer.zero_grad()
        logits = model(images=images, captions=caption_in)  # [B,T,V]

        # reshape for loss: merge batch and time
        loss = criterion(
            logits.view(-1, vocab_size),
            caption_label.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({'loss': loss.item(), 'epoch': epoch})
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} — Avg Loss: {avg_loss:.4f}")
    wandb.log({'epoch': epoch, 'train_loss': avg_loss})

# save model checkpoint
os.makedirs('checkpoints', exist_ok=True)
checkpoint_path = 'checkpoints/clip_caption_model_local.pth'
torch.save(model.state_dict(), checkpoint_path)
print(f"Model checkpoint saved at {checkpoint_path}")

# log wandb artifact
artifact = wandb.Artifact(f'week4f_lickr30k_{timestamp}', type='model')
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)

wandb.finish()