import os
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from caption_model import ImageCaptioningModel
from open_clip import get_tokenizer
from clip_utils import DEVICE
import datetime
from flickr_dataset import FlickrStreamDataset

#
#
# SETUP
#
#

BATCH_SIZE = 32
EPOCHS = 12
MAX_LEN = 32
LEARNING_RATE = 3e-4
SPLIT = 'test'
# got best results with 20k samples and 8 epochs - trying with full dataset and more epochs
SAMPLE_SIZE = 31783
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# initialize tokenizer and vocab size   
tokenizer = get_tokenizer('ViT-B-32')
sos_id = tokenizer.encoder.get('<start_of_text>', 49406)
eos_id = tokenizer.encoder.get('<end_of_text>', 49407)
pad_id = 0 
vocab_size= tokenizer.vocab_size

#
#
# DATA
# 
#

train_ds = FlickrStreamDataset(SPLIT, SAMPLE_SIZE)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

#
#
# model, optimizer, wandb
#
#

model = ImageCaptioningModel(decoder_vocab_size=vocab_size, decoder_max_len=MAX_LEN).to(DEVICE)
print(f"Model initialized on {DEVICE} with vocab size {vocab_size} and max len {MAX_LEN} with {SAMPLE_SIZE} samples.")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

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

#
#
# TRAINING LOOP
#
#

for epoch in range(1, EPOCHS + 1):
    model.train()
    # training loop
    total_loss = 0.0
    steps = 0
    progress_bar = tqdm(train_loader, total=SAMPLE_SIZE//BATCH_SIZE, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for images, caption_in, caption_label in progress_bar:
        images = images.to(DEVICE, non_blocking=True)
        caption_in = caption_in.to(DEVICE)
        caption_label = caption_label.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images=images, captions=caption_in)

        loss = criterion(
            logits.view(-1, vocab_size),
            caption_label.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        wandb.log({'loss': loss.item(), 'epoch': epoch})
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / steps
    print(f"Epoch {epoch} — Average Loss: {avg_loss:.4f}")
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