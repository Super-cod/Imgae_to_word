# 📷 Image Captioning with PyTorch

This project implements an image captioning model using a CNN-RNN architecture with PyTorch. The model is trained on the **Flickr30k** dataset to generate descriptive captions for given images.

---

## 📌 Description

The model uses a pre-trained **ResNet-50** as a CNN encoder to extract feature vectors from images. These features are then fed into a **Long Short-Term Memory (LSTM)** network decoder to generate a sequence of words that form a caption.

- **Encoder**: ResNet-50 (pre-trained on ImageNet)
- **Decoder**: LSTM Network
- **Dataset**: Flickr30k
- **Framework**: PyTorch

---

## 📁 Project Structure

```
image_captioning/
├── data/
│   ├── flickr30k_images/
│   │   └── ... (image files)
│   └── cleaned_results.csv
├── saved_models/
│   └── final_image_captioning_model.pth
├── config.py
├── dataset.py
├── model.py
├── vocabulary.py
├── train.py
├── test.py
└── README.md
```

---

## ⚙️ Setup and Installation

Follow these steps to set up your environment and install dependencies.

### 1. Clone the repository

```bash
git clone <repo_url>
cd image_captioning
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` with:

```
torch
torchvision
pandas
spacy
tqdm
Pillow
```

Then run:

```bash
pip install -r requirements.txt
```

### 4. Download SpaCy English model

```bash
python -m spacy download en_core_web_sm
```

---

## 🗂️ Dataset

This model is trained on the **Flickr30k** dataset.

- Download the dataset and place all images in:  
  `data/flickr30k_images/`

- Ensure your captions CSV is named:  
  `data/cleaned_results.csv`

---

## 🚀 Usage

### 🏋️‍♂️ Training the Model

Run the training script:

```bash
python train.py
```

All training hyperparameters and paths can be modified in `config.py`.

The trained model and vocabulary object will be saved to:

```
saved_models/final_image_captioning_model.pth
```

---

### 🧠 Generating a Caption (Testing)

To generate a caption for an image:

```bash
python test.py "path/to/your/image.jpg"
```

Example:

```bash
python test.py "data/flickr30k_images/69854977.jpg"
```

The script will:
- Load the model
- Generate a caption
- Print the caption
- Attempt to display the image

---

## 📄 File Breakdown

- `config.py`: Contains hyperparameters and path configs.
- `vocabulary.py`: Handles tokenization and word-to-index mapping.
- `dataset.py`: Loads images and captions via `FlickerDataset` and `MyCollate`.
- `model.py`: Defines `EncoderCNN` and `DecoderRNN` (ResNet + LSTM).
- `train.py`: Script to train the model and save weights.
- `test.py`: Loads a trained model and generates a caption for a given image.

---

## 📬 License & Contributions

Feel free to fork and improve this project. Contributions are welcome!
