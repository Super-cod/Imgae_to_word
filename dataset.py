import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from PIL import Image
import os

class FlickerDataset(Dataset):
    def __init__(self, root_dir, caption_file, vocab, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        self.vocab = vocab
        self.images = self.df["images"]
        self.captions = self.df["captions"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = os.path.join(self.root_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions