#!/usr/bin/env python3
"""
Setup script for the backend
"""
import os
import sys
import subprocess

def setup_backend():
    """Setup the backend environment"""
    print("Setting up Cucumber Disease Detection Backend...")
    
    # Create necessary directories
    directories = [
        'static/uploads',
        'models',
        'datasets',
        'results',
        'utils',
        'routes'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        sys.exit(1)
    
    # Install Python dependencies
    print("\nInstalling Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    print("\n✅ Setup complete!")
    print("\nTo start the backend server:")
    print("  python app.py")
    print("\nThe server will run at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /health         - Health check")
    print("  POST /api/detect     - Detect disease from image")
    print("  POST /api/batch-detect - Batch process images")
    print("  GET  /api/system-info - System information")

if __name__ == '__main__':
    setup_backend()