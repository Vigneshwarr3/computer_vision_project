import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Select 10 classes from ImageNet
# Using common classes: n01440764 (tench), n02124075 (Egyptian cat), n02834778 (bicycle), 
# n03028079 (church), n03394916 (French horn), n03417042 (garbage truck), 
# n03425413 (gas pump), n03445777 (golf ball), n03888257 (parachute), n04259630 (soccer ball)
SELECTED_CLASSES = [
    'n01440764', 'n02124075', 'n02834778', 'n03028079', 'n03394916',
    'n03417042', 'n03425413', 'n03445777', 'n03888257', 'n04259630'
]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
  
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # For CIFAR-10: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 = 256*2*2 = 1024
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size=32, data_dir='./data'):

    # CIFAR-10 has exactly 10 classes
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, test_loader, class_names


def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.001):
    """Train the model and return training history"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.2f}%'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*val_correct/val_total:.2f}%'})
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  -> Saved best model (Val Acc: {val_acc:.2f}%)')
    
    return history

def plot_training_curves(history):
    """Plot training and validation curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print('Training curves saved to training_curves.png')
    plt.show()

def visualize_predictions(model, test_loader, class_names, num_images=10):
    """Visualize predictions on 10 random images"""
    model.eval()
    
    # Get a batch of images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Move to device
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Move back to CPU for visualization
    images = images.cpu()
    predicted = predicted.cpu()
    probabilities = probabilities.cpu()
    
    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx in range(num_images):
        img = images[idx]
        # Denormalize
        img = img.permute(1, 2, 0)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        true_label = class_names[labels[idx]]
        pred_label = class_names[predicted[idx]]
        confidence = probabilities[idx][predicted[idx]].item() * 100
        
        color = 'green' if labels[idx] == predicted[idx] else 'red'
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                           color=color, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    print('Predictions visualization saved to predictions.png')
    plt.show()

def main():
    print("=" * 60)
    print("CNN Model Training on 10-Class Dataset")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading dataset...")
    train_loader, test_loader, class_names = get_data_loaders(batch_size=32)
    print(f"Dataset loaded! Classes: {class_names}")
    print()
    
    # Create model
    print("Creating CNN model...")
    model = SimpleCNN(num_classes=10).to(device)
    print(f"Model created! Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001)
    print()
    
    # Load best model
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load('best_model.pth'))
    print()
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(history)
    print()
    
    # Visualize predictions
    print("Visualizing predictions on 10 images...")
    visualize_predictions(model, test_loader, class_names, num_images=10)
    print()
    
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

