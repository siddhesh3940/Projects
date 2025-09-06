#!/usr/bin/env python3

import subprocess
import sys
import os

def install_simple_requirements():
    """Install minimal requirements"""
    print("Installing minimal Python packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "simple_requirements.txt"])
        print("✅ Basic packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'known_faces']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def run_server():
    """Run the Flask server"""
    print("\n🚀 Starting Face Recognition Server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("\n⚠️  Note: Using simplified mode (mock face recognition)")
    print("💡 To enable real face recognition, install: pip install opencv-python face-recognition dlib")
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Make sure app.py is in the current directory")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    print("🔧 Setting up Simple Face Recognition System...")
    
    create_directories()
    
    if install_simple_requirements():
        print("\n✅ Setup complete!")
        run_server()
    else:
        print("\n❌ Setup failed. Please install packages manually:")
        print("pip install Flask flask-cors Pillow numpy")
        print("Then run: python app.py")