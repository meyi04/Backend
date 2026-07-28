import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
import logging
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage trained cucumber disease detection models"""
    
    @staticmethod
    def load_model(model_path):
        """
        Load a trained PyTorch model
        """
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model architecture
            num_classes = checkpoint.get('num_classes', 6)  # Default to 6 classes
            
            # Use the same architecture as training (ResNet18 + custom head)
            model = models.resnet18(pretrained=False)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get class names
            class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
            
            logger.info(f"Model loaded successfully: {model_path}")
            logger.info(f"Classes: {class_names}")
            
            return {
                'model': model,
                'class_names': class_names,
                'num_classes': num_classes,
                'image_size': checkpoint.get('image_size', (224, 224)),
                'metadata': checkpoint
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @staticmethod
    def load_tensorflow_model(model_path):
        """Load TensorFlow/Keras model"""
        try:
            import tensorflow as tf
            
            model = tf.keras.models.load_model(model_path)
            
            # Try to get class names from model or config
            class_names = None
            if hasattr(model, 'class_names'):
                class_names = model.class_names
            
            return {
                'model': model,
                'class_names': class_names,
                'type': 'tensorflow'
            }
            
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            raise
    
    @staticmethod
    def get_available_models():
        """List all trained models"""
        models_dir = Config.MODELS_DIR
        
        if not models_dir.exists():
            return []
        
        models = []
        for model_file in models_dir.iterdir():
            if model_file.suffix.lower() in ['.pth', '.pt', '.h5', '.keras']:
                models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'size': model_file.stat().st_size,
                    'type': model_file.suffix[1:]  # Remove dot
                })
        
        return models
