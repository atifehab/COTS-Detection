# COTS Detector — Crown-of-Thorns Starfish Classification

A deep learning project for binary classification of Crown-of-Thorns Starfish (COTS) in underwater images, using a fine-tuned **EfficientNet-B0** model deployed via **Streamlit**.

The Crown-of-Thorns Starfish (*Acanthaster planci*) is one of the most destructive predators of coral reefs. Early and accurate detection is critical for marine conservation efforts. This project provides both the training pipeline and a web-based inference tool for real-time COTS classification.

---

## Project Structure

```
NTI project/
├── app.py                          # Streamlit deployment app (web UI)
├── requirements.txt                # Python dependencies
├── src/
│   └── EfficientNet_train.py       # Model training script
└── cots_model/                     # Saved model weights (state_dict)
    ├── data.pkl
    ├── data/
    └── ...
```

---

## Model Architecture

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| **Backbone**      | EfficientNet-B0 (pretrained on ImageNet)    |
| **Classifier**    | Linear(1280, 2)                             |
| **Classes**       | Class 0: COTS / Class 1: Not COTS          |
| **Input Size**    | 224 x 224 RGB                               |
| **Optimizer**     | Adam (lr=0.001)                             |
| **Loss Function** | CrossEntropyLoss                            |
| **Epochs**        | 20                                          |
| **Data Split**    | 70% Train / 15% Validation / 15% Test      |

---

## Dataset

**Binary Cropped Crown-of-Thorns Dataset** from Kaggle:

[https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset](https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset)

The dataset contains cropped underwater images organized into two folders:
- `cots_crops/` — Images containing COTS
- `notcots_crops/` — Images without COTS

---

## Training

The training script (`src/EfficientNet_train.py`) performs the following steps:

1. **Downloads** the dataset from Kaggle using the `opendatasets` library
2. **Splits** the data into train (70%), validation (15%), and test (15%) sets
3. **Applies** data augmentation (random horizontal flip, random rotation) on the training set
4. **Fine-tunes** EfficientNet-B0 with ImageNet pretrained weights for 20 epochs
5. **Evaluates** on the test set and prints accuracy, precision, recall, F1-score, and confusion matrix
6. **Saves** the best model weights (by validation accuracy) to disk

### Preprocessing

All images are resized to 224x224 and normalized with ImageNet statistics:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

---

## Deployment

The Streamlit app (`app.py`) provides a web interface for real-time inference:

### Features
- Upload images (JPG, JPEG, PNG) via drag-and-drop
- Capture images directly from webcam
- View prediction result with confidence score
- See probability breakdown for both classes

### Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### How It Works

1. The saved EfficientNet-B0 state_dict is loaded and cached in memory
2. Uploaded images are preprocessed (resize, crop, normalize)
3. The model runs a forward pass and applies softmax to get class probabilities
4. Results are displayed with a confidence bar and class breakdown

---

## Installation

```bash
git clone https://github.com/<your-username>/Computer-vision-project.git
cd Computer-vision-project
pip install -r requirements.txt
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- Streamlit 1.28+
- Pillow 9.0+
- scikit-learn 1.0+

See `requirements.txt` for the full list.

---

## Technologies Used

- **PyTorch** — Model architecture, training, and inference
- **torchvision** — EfficientNet-B0 backbone and image transforms
- **Streamlit** — Web application framework for deployment
- **scikit-learn** — Evaluation metrics (accuracy, precision, recall, F1)
- **opendatasets** — Kaggle dataset download automation
