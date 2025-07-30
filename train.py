import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

import config
from vocabulary import Vocabulary
from dataset import FlickerDataset, MyCollate
from model import EncoderCNN, DecoderRNN

def train():
    print("Loading data and building vocabulary...")
    df = pd.read_csv(config.CAPTION_FILE)
    df.columns = df.columns.str.strip()
    captions_list = df["captions"].dropna().astype(str).tolist()

    vocab = Vocabulary(config.FREQ_THRESHOLD)
    vocab.build_vocabulary(captions_list)
    pad_idx = vocab.stoi["<PAD>"]

    dataset = FlickerDataset(
        root_dir=config.ROOT_DIR,
        caption_file=config.CAPTION_FILE,
        vocab=vocab,
        transform=config.IMAGE_TRANSFORMS_TRAIN
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    vocab_size = len(vocab)

    print(f"Training on device: {config.DEVICE}")

    encoder = EncoderCNN(config.EMBED_SIZE).to(config.DEVICE)
    decoder = DecoderRNN(config.EMBED_SIZE, config.HIDDEN_SIZE, vocab_size, config.NUM_LAYERS).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE)

    print("Starting training...")
    encoder.train()
    decoder.train()

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
        for idx, (imgs, captions) in loop:
            imgs = imgs.to(config.DEVICE)
            captions = captions.to(config.DEVICE)

            features = encoder(imgs)
            outputs = decoder(features, captions)

            loss = criterion(
                outputs[:, :-1, :].reshape(-1, vocab_size),
                captions[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

    print("Training finished. Saving model...")

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'vocab': vocab,
    }, config.MODEL_PATH)

    print(f"Model saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    train()