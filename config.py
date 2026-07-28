import os
from pathlib import Path
from dotenv import load_dotenv

# ⬇️ ADD THIS LINE - Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    # Now this will read from .env file first, then fallback
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # ⬇️ ADD THIS - OpenAI API Key
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GPT_MODEL = os.environ.get('GPT_MODEL', 'gpt-4-turbo')
    
    DEBUG = True
    
    # File upload configuration
    BASE_DIR = Path(__file__).parent.absolute()
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Model configuration
    MODELS_DIR = BASE_DIR / 'models' / 'trained'
    
    # Dataset paths
    DATASETS_DIR = BASE_DIR / 'datasets'
    RAW_DATA_DIR = DATASETS_DIR / 'raw'
    PROCESSED_DATA_DIR = DATASETS_DIR / 'processed'
    
    # Default image size for model input
    IMAGE_SIZE = (224, 224)
    
    # Default class names (update with your actual classes)
    CLASS_NAMES = [
        'healthy',
        'angular_leaf_spot', 
        'downy_mildew',
        'anthracnose',
        'powdery_mildew',
        'bacterial_wilt'
    ]
    
    @staticmethod
    def init_app():
        """Initialize application directories"""
        # Create necessary directories
        directories = [
            Config.UPLOAD_FOLDER,
            Config.MODELS_DIR,
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is set"""
        missing = []
        
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        
        if missing:
            print(f"⚠️  Warning: Missing environment variables: {missing}")
            print("   GPT features will be disabled.")
        
        return len(missing) == 0