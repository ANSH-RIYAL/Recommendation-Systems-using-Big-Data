#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'logs',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_virtual_environment():
    """Create and activate virtual environment."""
    if not os.path.exists('venv'):
        subprocess.run(['python', '-m', 'venv', 'venv'])
        print("Created virtual environment: venv")
    
    # Install requirements
    pip_cmd = 'venv/bin/pip' if os.name != 'nt' else 'venv\\Scripts\\pip'
    subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'])
    print("Installed project dependencies")

def setup_config():
    """Create config file from example if it doesn't exist."""
    if not os.path.exists('config/config.yaml'):
        shutil.copy('config/config.example.yaml', 'config/config.yaml')
        print("Created config file from example")

def main():
    print("Setting up Recommendation System project...")
    
    # Create directories
    create_directories()
    
    # Setup virtual environment
    setup_virtual_environment()
    
    # Setup configuration
    setup_config()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print("   - On Unix/macOS: source venv/bin/activate")
    print("   - On Windows: venv\\Scripts\\activate")
    print("2. Place your data files in the data/raw directory")
    print("3. Configure your settings in config/config.yaml")
    print("4. Run the preprocessing pipeline:")
    print("   python src/data/preprocessing.py")
    print("5. Train and evaluate models:")
    print("   python src/models/train.py")

if __name__ == "__main__":
    main() 