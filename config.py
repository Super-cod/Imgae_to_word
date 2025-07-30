import torch
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = "data/flickr30k_images"
CAPTION_FILE = "data/cleaned_results.csv"
MODEL_PATH = "saved_models/final_image_captioning_model.pth"

FREQ_THRESHOLD = 10

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 3

LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
BATCH_SIZE = 32

IMAGE_TRANSFORMS_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IMAGE_TRANSFORMS_TEST = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])