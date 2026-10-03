# Deep Learning Projects (PyTorch)

A collection of deep learning mini projects built with PyTorch.

## 1. Cats vs Dogs Image Classifier (CNN)
Notebook: `image_classification0.ipynb`

Binary image classifier that predicts whether an image is a cat or a dog,
using a custom CNN trained on the Microsoft Cats vs Dogs dataset.

- Cleaned the dataset by removing corrupt images (~11,200 valid images)
- Preprocessing: resize to 128x128, normalization
- 70% train / 15% validation / 15% test split
- Model: 3 convolutional layers + MaxPooling + 2 fully connected layers
- Per-epoch validation with best-model checkpointing
- Evaluated with accuracy, precision, recall and confusion matrix

| Metric | Score |
|---|---|
| Accuracy | 77.65% |
| Precision | 0.7721 |
| Recall | 0.7366 |

## 2. CIFAR-10 Image Classifier (CNN)
Notebook: `CNN_for_CIFAR10-2.ipynb`

10-class image classifier (airplane, car, bird, cat, etc.) built with a CNN
on the CIFAR-10 dataset.

- 3 convolutional layers + MaxPooling + fully connected layers
- Adam optimizer, CrossEntropyLoss, 10 epochs

| Metric | Score |
|---|---|
| Test Accuracy | 76.16% |

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, Matplotlib, Jupyter
