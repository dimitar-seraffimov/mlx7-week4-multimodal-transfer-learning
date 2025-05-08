import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from open_clip import get_tokenizer
from tqdm import tqdm

class Flickr30kCaptionDataset(Dataset):
    def __init__(self,
                 split='test',
                 max_caption_len=32,
                 image_size=224,
                 sample_size=500):
        super().__init__()
        raw_dataset = load_dataset("nlphuji/flickr30k", split=split, streaming=True)

        self.raw_stream = iter(raw_dataset)
        self.samples = []
        self.sample_size = sample_size

        self.split = split
        self.max_caption_len = max_caption_len
        self.image_size = image_size

        self.tokenizer = get_tokenizer('ViT-B-32')
        self.sos_id = self.tokenizer.encoder.get('<start_of_text>', 0)
        self.eos_id = self.tokenizer.encoder.get('<end_of_text>', 0)
        self.pad_id = 0

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        print("Preparing streamed dataset...")
        count = 0

        for row in tqdm(self.raw_stream, desc="Streaming batch", total=self.sample_size):
            img=row['image']
            for caption in row['caption']:
                token_ids = self.tokenizer([caption])[0].tolist()
                token_ids = token_ids[:self.max_caption_len - 2]  # reserve space for SOS and EOS

                input_ids = [self.sos_id] + token_ids
                label_ids = token_ids + [self.eos_id]

                input_ids += [self.pad_id] * (self.max_caption_len - len(input_ids))
                label_ids += [self.pad_id] * (self.max_caption_len - len(label_ids))

                self.samples.append({
                    'image': img,
                    'caption_input': input_ids,
                    'caption_label': label_ids
                })

                count += 1
                if count >= self.sample_size:
                    break
            if count >= self.sample_size:
                break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = sample['image'].convert('RGB')
        img_tensor = self.image_transform(img)
        input_tensor = torch.tensor(sample['caption_input'], dtype=torch.long)
        label_tensor = torch.tensor(sample['caption_label'], dtype=torch.long)

        return img_tensor, input_tensor, label_tensor
    
if __name__ == "__main__":
    dataset = Flickr30kCaptionDataset(split='test')
    print(f"Loaded {len(dataset)} samples")
    img, input_ids, label_ids = dataset[0]
    print(f"Shapes: image {img.shape}, input_ids {input_ids.shape}, label_ids {label_ids.shape}")