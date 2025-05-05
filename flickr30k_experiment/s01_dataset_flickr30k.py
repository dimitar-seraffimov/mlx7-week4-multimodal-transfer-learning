import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from open_clip import get_tokenizer
from tqdm import tqdm

class Flickr30kCaptionDataset(Dataset):
    def __init__(self,
                 split='test',
                 max_caption_len=30,
                 image_size=224):
        super().__init__()
        # Load nlphuji Flickr30k (only 'test' split available)
        ds = load_dataset("nlphuji/flickr30k", split=split)

        # CLIP tokenizer for captions
        self.tokenizer = get_tokenizer('ViT-B-32')
        self.bos_id = self.tokenizer.encoder.get('<|startoftext|>', 0)
        self.eos_id = self.tokenizer.encoder.get('<|endoftext|>', 0)
        self.pad_id = 0  # Use 0 as pad

        self.max_caption_len = max_caption_len
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.samples = []
        self._prepare_samples(ds)

    def _prepare_samples(self, ds):
        print("Preparing (image_tensor, caption_input, caption_label) from nlphuji/flickr30k...")
        for row in tqdm(ds, total=len(ds)):
            # Load image (datasets.Image returns PIL.Image)
            img_info = row['image']
            if isinstance(img_info, dict) and 'path' in img_info:
                img = Image.open(img_info['path']).convert('RGB')
            else:
                img = img_info.convert('RGB')
            img_t = self.image_transform(img)

            # 'caption' field: list of strings
            for raw in row['caption']:
                ids = self.tokenizer([raw])[0].tolist()

                # prepare input and label sequences
                inp = [self.bos_id] + ids
                tgt = ids + [self.eos_id]

                # truncate
                inp = inp[:self.max_caption_len]
                tgt = tgt[:self.max_caption_len]
                # pad
                inp += [self.pad_id] * (self.max_caption_len - len(inp))
                tgt += [self.pad_id] * (self.max_caption_len - len(tgt))

                self.samples.append((img_t, torch.tensor(inp), torch.tensor(tgt)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == '__main__':
    ds = Flickr30kCaptionDataset(split='test', max_caption_len=30)
    print(f"Loaded {len(ds)} samples")
    img, inp, tgt = ds[0]
    print(f"Shapes: image {img.shape}, input {inp.shape}, target {tgt.shape}")
