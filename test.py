import torch
from PIL import Image
import argparse

import config
from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary


def generate_caption(image_path, encoder, decoder, vocab, transform, device, max_length=50):
    encoder.eval()
    decoder.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    result_caption = []

    with torch.no_grad():
        features = encoder(image)
        inputs = features.unsqueeze(1)
        states = None

        for _ in range(max_length):
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            predicted_idx = outputs.argmax(1)

            result_caption.append(predicted_idx.item())

            if predicted_idx.item() == vocab.stoi["<EOS>"]:
                break

            inputs = decoder.embed(predicted_idx).unsqueeze(1)

    return ' '.join([vocab.itos[idx] for idx in result_caption if vocab.itos[idx] not in ["<SOS>", "<EOS>", "<PAD>"]])


def main(image_path):
    print("Loading model and vocabulary...")

    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)

    encoder = EncoderCNN(config.EMBED_SIZE).to(config.DEVICE)
    decoder = DecoderRNN(config.EMBED_SIZE, config.HIDDEN_SIZE, vocab_size, config.NUM_LAYERS).to(config.DEVICE)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print("Model loaded successfully!")

    caption = generate_caption(
        image_path,
        encoder,
        decoder,
        vocab,
        config.IMAGE_TRANSFORMS_TEST,
        config.DEVICE
    )

    print("\n--- Result ---")
    print(f"Generated Caption: {caption}")

    try:
        img = Image.open(image_path)
        img.show(title="Test Image")
    except Exception as e:
        print(f"\nCould not display image. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument("image_path", type=str, help="Path to the test image.")
    args = parser.parse_args()

    main(args.image_path)