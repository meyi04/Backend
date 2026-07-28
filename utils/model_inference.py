import logging
from pathlib import Path
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image

from config import Config
from utils.model_loader import ModelLoader
from models.placeholder_model import PlaceholderModel

logger = logging.getLogger(__name__)


class TrainedModelService:
    """Load the latest trained model and run inference."""

    _model_data = None
    _model_path = None

    @classmethod
    def _select_latest_model_path(cls):
        models = ModelLoader.get_available_models()
        if not models:
            return None
        latest = max(models, key=lambda m: Path(m['path']).stat().st_mtime)
        return latest['path']

    @classmethod
    def load(cls, force=False):
        if cls._model_data is not None and not force:
            return cls._model_data

        model_path = cls._select_latest_model_path()
        if not model_path:
            logger.warning("No trained models found in %s", Config.MODELS_DIR)
            cls._model_data = None
            cls._model_path = None
            return None

        cls._model_path = model_path
        cls._model_data = ModelLoader.load_model(model_path)
        return cls._model_data

    @classmethod
    def predict(cls, image_path):
        try:
            model_data = cls.load()
            if model_data is None:
                return {'success': False, 'error': 'No trained model available'}

            model = model_data['model']
            class_names = model_data.get('class_names') or Config.CLASS_NAMES
            image_size = model_data.get('image_size', Config.IMAGE_SIZE)

            transform = transforms.Compose([
                transforms.Resize(tuple(image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

            top_idx = int(probs.argmax())
            top_class = class_names[top_idx] if top_idx < len(class_names) else f'class_{top_idx}'
            top_confidence = float(probs[top_idx])

            all_predictions = {
                class_names[i] if i < len(class_names) else f'class_{i}': float(probs[i])
                for i in range(len(probs))
            }

            suggestions = []
            if hasattr(PlaceholderModel, '_get_suggestions'):
                suggestions = PlaceholderModel._get_suggestions(top_class)

            return {
                'success': True,
                'prediction': {
                    'disease': top_class,
                    'confidence': top_confidence,
                    'all_predictions': all_predictions,
                    'suggestions': suggestions,
                    'timestamp': datetime.now().isoformat(),
                    'model_path': cls._model_path
                }
            }

        except Exception as e:
            logger.exception("Model inference failed")
            return {'success': False, 'error': str(e)}
