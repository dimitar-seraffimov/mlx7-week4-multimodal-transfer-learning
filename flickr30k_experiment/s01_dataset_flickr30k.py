import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from open_clip import get_tokenizer
from tqdm import tqdm
import os

class Flickr30kCaptionDataset(Dataset):
    def __init__(self,
                 split='test',
                 max_caption_len=30,
                 image_size=224,
                 cache_file='flickr30k_cache.pt'):
        super().__init__()
        # fetch the dataset metadata from HuggingFace nlphuji Flickr30k (only 'test' split available)
        # - "nlphuji/flickr30k" has only 'test' split
        # - load_dataset returns a list-like object of rows
        ds = load_dataset("nlphuji/flickr30k", split=split)

        self.split = split
        self.max_caption_len = max_caption_len
        self.image_size = image_size
        self.cache_file = cache_file

        # CLIP tokenizer for captions
        self.tokenizer = get_tokenizer('ViT-B-32')
        self.bos_id = self.tokenizer.encoder.get('<|startoftext|>', 0)
        self.eos_id = self.tokenizer.encoder.get('<|endoftext|>', 0)
        self.pad_id = 0  # use 0 as pad

        # define image transform - store params for later use
        # - resize to 224x224
        # - convert to tensor
        # - normalize to [-1, 1]
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # load or build the list of triplets
        if os.path.exists(self.cache_file):
            # load cached metadata (small file)
            print(f"Loading cached triplets from {self.cache_file}...")
            self.samples = torch.load(self.cache_file)
        else:
            # build from raw dataset
            print("Building and caching triplets for the first time...")
            self.samples = []
            raw = load_dataset("nlphuji/flickr30k", split=split)
            for row in tqdm(raw, total=len(raw)):
                img_info = row['image']
                if isinstance(img_info, dict) and 'path' in img_info:
                    img_path = img_info['path']
                    print(f"Image path: {img_path} Type: {type(img_info)}")
                elif isinstance(img_info, str):
                    img_path = img_info
                    print(f"Image path: {img_path} Type: {type(img_info)}")
                elif hasattr(img_info, 'path'):
                    img_path = img_info.path
                    print(f"Image path: {img_path} Type: {type(img_info)}")
                else:
                    raise ValueError(f"Cannot extract image path from type: {type(img_info)}")

                # tokenize and produce one sample per caption
                for raw_caption in row['caption']:
                    token_ids = self.tokenizer([raw_caption])[0].tolist()
                    # prepare input and target sequences
                    input_ids = [self.bos_id] + token_ids
                    label_ids = token_ids + [self.eos_id]
                    # truncate and pad to fixed length
                    input_ids = input_ids[:self.max_caption_len] + [self.pad_id] * (self.max_caption_len - len(input_ids))
                    label_ids = label_ids[:self.max_caption_len] + [self.pad_id] * (self.max_caption_len - len(label_ids))
                    # store metadata only
                    self.samples.append({
                        'image_path': img_path,
                        'caption_in': input_ids,
                        'caption_label': label_ids
                    })

        # save to disk
        torch.save(self.samples, self.cache_file)
        print(f"Saved {len(self.samples)} triplets to {self.cache_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        # load and transform image on-the-fly
        img = Image.open(record['image_path']).convert('RGB')
        img_t = self.image_transform(img)
        # convert token lists to tensors
        input_t = torch.tensor(record['caption_in'], dtype=torch.long)
        label_t = torch.tensor(record['caption_label'], dtype=torch.long)
        return img_t, input_t, label_t

if __name__ == '__main__':
    ds = Flickr30kCaptionDataset(split='test')
    print(f"Loaded {len(ds)} samples")
    img, input, label = ds[0]
    print(f"Shapes: image {img.shape}, input {input.shape}, label {label.shape}")
