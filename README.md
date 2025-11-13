# Image Classification Models for 10-Class Dataset

This project implements both a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) for image classification using a 10-class dataset. Both models are trained on CIFAR-10 dataset (which has exactly 10 classes) as a practical alternative to ImageNet subset.

## Features

- **CNN Architecture**: Custom CNN with 2 convolutional layers, batch normalization, and dropout
- **Vision Transformer**: Full ViT implementation with patch embedding, multi-head self-attention, and transformer blocks
- **Training & Validation Curves**: Automatic plotting of loss and accuracy curves for both models
- **Prediction Visualization**: Displays predictions on 10 test images with confidence scores
- **Model Checkpointing**: Saves the best model based on validation accuracy

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- tqdm

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Train CNN Model
Run the CNN training script:
```bash
python train_cnn.py
```

### Train Vision Transformer Model
Run the ViT training script:
```bash
python train_vit.py
```

Both scripts will:
1. Download the CIFAR-10 dataset (if not already present)
2. Train the model for 20 epochs
3. Save the best model weights
4. Generate training/validation curves
5. Generate predictions visualization on 10 test images

## Model Architectures

### CNN Model
- **Convolutional Layers**: 2 layers with 32 and 64 filters
- **Batch Normalization**: Applied after each convolutional layer
- **Max Pooling**: 2x2 pooling after each conv layer
- **Dropout**: 0.5 dropout rate before final classification
- **Fully Connected Layers**: 512 hidden units, 10 output classes

### Vision Transformer Model
- **Patch Embedding**: Splits 32x32 images into 4x4 patches (64 patches total)
- **Embedding Dimension**: 128
- **Transformer Depth**: 6 transformer blocks
- **Attention Heads**: 8 multi-head self-attention heads
- **MLP Ratio**: 4x expansion in feed-forward networks
- **Classification Head**: Linear layer for 10 classes

## Output Files

### CNN Model
- `best_model.pth`: Saved CNN model weights
- `training_curves.png`: Training and validation curves
- `predictions.png`: Visualizations of 10 test image predictions

### Vision Transformer Model
- `best_vit_model.pth`: Saved ViT model weights
- `vit_training_curves.png`: Training and validation curves
- `vit_predictions.png`: Visualizations of 10 test image predictions

## Dataset

The model uses CIFAR-10 dataset which contains 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Each image is 32x32 pixels with 3 color channels.

## Notes

- The model automatically uses GPU if available, otherwise falls back to CPU
- Training progress is displayed with progress bars
- The best model (highest validation accuracy) is automatically saved

