import os
import torch
from torch.utils.data import IterableDataset, DataLoader
import wandb
from tqdm import tqdm
from caption_model import ImageCaptioningModel
from open_clip import get_tokenizer
from clip_utils import DEVICE
from datasets import load_dataset
from torchvision import transforms
import datetime

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
SAMPLE_SIZE = 5000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# initialize tokenizer and vocab size   
tokenizer = get_tokenizer('ViT-B-32')
pad_id    = 0
vocab_size = tokenizer.vocab_size

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

#
#
# STREAMED DATASET
#
#

class FlickrCaptionIter(IterableDataset):
    def __init__(self, split, max_len, sample_size):
        ds = load_dataset("nlphuji/flickr30k", split=split, streaming=True)
        self.stream = iter(ds.shuffle(buffer_size=1000))
        self.max_len = max_len
        self.sample_size = sample_size

    def __iter__(self):
        count = 0
        for row in self.stream:
            img = row['image'].convert('RGB')
            img_t = image_transform(img)
            for cap in row['caption']:
                tokens = tokenizer([cap])[0].tolist()
                seq = tokens[:self.max_len]
                padded = seq + [pad_id] * (self.max_len - len(seq))
                # input and label are the same
                yield img_t, torch.tensor(padded), torch.tensor(padded)
                count += 1
                if count>=self.sample_size:
                    return

#
#
# DATA
# 
#


train_ds = FlickrCaptionIter(SPLIT, MAX_LEN, SAMPLE_SIZE)
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

model = ImageCaptioningModel(vocab_size, MAX_LEN).to(DEVICE)
print(f"Model initialized on {DEVICE} with vocab size {vocab_size} and max len {MAX_LEN}")
optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion  = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

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
    total_loss = 0.0
    progress_bar = tqdm(train_loader, total=SAMPLE_SIZE//BATCH_SIZE, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for images, caption_in, caption_label in progress_bar:
        images = images.to(DEVICE, non_blocking=True)           # [B,3,H,W]
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
        progress_bar.set_postfix({'loss':f"{loss.item():.4f}"})

    avg_loss = total_loss / (SAMPLE_SIZE//BATCH_SIZE)
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