import os
from werkzeug.utils import secure_filename
from datetime import datetime
from config import Config
import shutil

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file with timestamp"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        return filepath, unique_filename
    return None, None

def cleanup_old_files(hours=24):
    """Clean up files older than specified hours"""
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=hours)
    
    if not os.path.exists(Config.UPLOAD_FOLDER):
        return
    
    for filename in os.listdir(Config.UPLOAD_FOLDER):
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        try:
            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
            if file_time < cutoff_time:
                os.remove(filepath)
        except:
            pass

def organize_dataset(source_dir, target_dir):
    """
    Organize dataset into class folders
    """
    import glob
    
    # Create target directory structure
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    # Organize by class (you need to implement your own logic)
    # For now, just copy all images to a single folder
    processed_dir = os.path.join(target_dir, 'all_images')
    os.makedirs(processed_dir, exist_ok=True)
    
    for img_file in image_files:
        filename = os.path.basename(img_file)
        dest = os.path.join(processed_dir, filename)
        shutil.copy2(img_file, dest)
    
    return len(image_files)