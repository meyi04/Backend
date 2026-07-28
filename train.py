#!/usr/bin/env python3
"""
Training script for cucumber disease detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from config import Config
import cv2
import os

class CucumberDataset(Dataset):
    """Dataset for cucumber disease images"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find class directories
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load image paths
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.image_paths.append(str(img_file))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model():
    """Train the cucumber disease detection model"""
    print("🚀 Starting model training...")
    
    # Check if dataset exists
    train_dir = Config.RAW_DATA_DIR / 'train'
    val_dir = Config.RAW_DATA_DIR / 'val'
    
    if not train_dir.exists():
        print(f"❌ Training data not found at: {train_dir}")
        print("Please organize your dataset in datasets/raw/train/ and datasets/raw/val/")
        return
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CucumberDataset(train_dir, transform=train_transform)
    val_dataset = CucumberDataset(val_dir, transform=val_transform)
    
    print(f"📊 Dataset loaded:")
    print(f"   Classes: {train_dataset.classes}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training images found!")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    best_val_acc = 0
    
    print(f"\n🎯 Training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_dataset)
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, train_dataset.classes, epoch+1, val_acc)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

def save_model(model, class_names, epoch, accuracy):
    """Save trained model"""
    # Create models directory
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare model data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"cucumber_disease_model_{timestamp}.pth"
    model_path = Config.MODELS_DIR / model_name
    
    # Save model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'image_size': (224, 224),
        'accuracy': accuracy,
        'timestamp': timestamp
    }, model_path)
    
    # Save model info
    model_info = {
        'name': model_name,
        'classes': class_names,
        'num_classes': len(class_names),
        'accuracy': float(accuracy),
        'image_size': [224, 224],
        'epoch': epoch,
        'timestamp': timestamp,
        'framework': 'pytorch'
    }
    
    info_path = Config.MODELS_DIR / f"{model_name}.info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"💾 Model saved: {model_path}")
    print(f"📝 Model info saved: {info_path}")
    
    return model_path

if __name__ == '__main__':
    train_model()