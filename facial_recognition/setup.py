import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing Python packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ['uploads', 'known_faces', 'templates', 'static']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def add_sample_faces():
    """Instructions for adding sample faces"""
    print("\n📸 To add known faces:")
    print("1. Place face images in the 'known_faces' folder")
    print("2. Name files as 'PersonName.jpg' (e.g., 'John_Doe.jpg')")
    print("3. Restart the application")

if __name__ == "__main__":
    print("🚀 Setting up Face Recognition System...")
    
    setup_directories()
    
    if install_requirements():
        add_sample_faces()
        print("\n✅ Setup complete! Run 'python app.py' to start the server.")
    else:
        print("\n❌ Setup failed. Please install packages manually.")