import torch
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset
from torchvision import transforms
from open_clip import get_tokenizer
from tqdm import tqdm

#
#
# GLOBAL SETUP
#
#

image_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize([0.5],[0.5])
])

tokenizer = get_tokenizer('ViT-B-32')
sos_id = tokenizer.encoder.get('<start_of_text>', 49406)
eos_id = tokenizer.encoder.get('<end_of_text>', 49407)
pad_id = 0 
vocab_size= tokenizer.vocab_size

MAX_LEN = 32
# got best results with 20k samples and 8 epochs
SAMPLE_SIZE = 20000

#
#
# helper function
#
#

def tokenize_caption(caption):
  caption = caption.rstrip('.')
  token_ids = tokenizer([caption])[0].tolist()
  token_ids = [t for t in token_ids if t not in {sos_id, eos_id}]
  token_ids = token_ids[:MAX_LEN - 2] # make room for SOS and EOS

  input_ids = [sos_id] + token_ids 
  label_ids = token_ids + [eos_id]

  input_ids += [pad_id] * (MAX_LEN - len(input_ids))
  label_ids += [pad_id] * (MAX_LEN - len(label_ids))

  return input_ids, label_ids

#
#
# DEBUG DATASET
#
#

class FlickrDebugDataset(Dataset):
    """
    Non-streaming dataset for debugging and evaluating only!
    Returns a tuple of (image_tensor, input_tensor, label_tensor) where:
        - image_tensor: Processed image tensor [3, 224, 224] ready for the CLIP encoder
        - input_tensor: Tokenized caption with SOS token for decoder input [MAX_LEN]
        - label_tensor: Target caption tokens for loss calculation [MAX_LEN]
    
    used in evaluation and testing when I need random access to samples
    """
    def __init__(self, split='test', sample_size=500):
        raw_dataset = load_dataset("nlphuji/flickr30k", split=split, streaming=True)
        self.samples = []
        count = 0
        
        progress_bar = tqdm(total=sample_size, desc="Loading Debug Samples")
        for row in raw_dataset:
            img = row['image']
            for caption in row['caption']:
                input_ids, label_ids = tokenize_caption(caption)
                self.samples.append({
                'image': img,
                'caption_input': input_ids,
                'caption_label': label_ids
                })
                count += 1
                progress_bar.update(1)
                if count >= sample_size:
                    break
            if count >= sample_size:
                break
        progress_bar.close()
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            # get sample
            image, input_ids, label_ids = self.samples[idx]

            # transform image
            image_tensor = image_transform(image.convert('RGB'))

            # convert to tensors the input and label ids
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)

            # return image, caption_input, caption_label
            return image_tensor, input_tensor, label_tensor    
    
#
#
# STREAM DATASET - TRAINING
#
#

class FlickrStreamDataset(IterableDataset):
    """  
    Streaming dataset used for training the image captioning model
    without loading everything into memory. Each iteration returns:
        - img_tensor: Processed image tensor ready for the CLIP encoder
        - input_ids: Tokenized caption with start token for decoder input
        - label_ids: Tokenized caption for computing loss during training
    """
    def __init__(self, split='test', sample_size=SAMPLE_SIZE):
        self.split = split
        self.sample_size = sample_size

    def __iter__(self):
        dataset = load_dataset("nlphuji/flickr30k", split=self.split, streaming=True)
        # streaming buffer size for each epoch loading the dataset
        buffer_size = 1000
        stream = iter(dataset.shuffle(buffer_size=buffer_size))
        # count samples
        count = 0
        for row in stream:
            # return image tensor
            img_tensor = image_transform(row['image'].convert('RGB'))
            # return caption input and label ids
            for caption in row['caption']:
                # check if the sample size limit is reached
                if count >= self.sample_size:
                    return
                input_ids, label_ids = tokenize_caption(caption)
                yield img_tensor, torch.tensor(input_ids), torch.tensor(label_ids)
                count += 1

#
#
# IMAGE-ONLY DATASET FOR CLIP EMBEDDING
#
#

class FlickrImageOnly(IterableDataset):
    """
    Streaming dataset that returns only processed image tensors from the Flickr30k dataset:
        - used only for generating CLIP image embeddings
        - each iteration returns a single preprocessed image tensor
    """
    def __init__(self, split='test', sample_size=SAMPLE_SIZE):
        self.split = split
        self.sample_size = sample_size

    def __iter__(self):
        dataset = load_dataset("nlphuji/flickr30k", split=self.split, streaming=True)
        stream = iter(dataset.shuffle(buffer_size=1000))
        progress_bar = tqdm(stream, total=self.sample_size, desc="Streaming Image Embeddings")

        for i, row in enumerate(progress_bar):
            if i >= self.sample_size:
                return
            yield image_transform(row['image'].convert('RGB'))