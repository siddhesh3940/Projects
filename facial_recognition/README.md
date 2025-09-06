# FaceRecog - AI Face Recognition System

A complete face recognition system with Python backend and modern web frontend.

## Features

- **Face Recognition**: Upload images or use live camera for face detection
- **Real-time Detection**: Live camera feed with face bounding boxes
- **Face Comparison**: Compare two faces side-by-side
- **User Authentication**: Login/signup with role-based access
- **Admin Panel**: Add known faces to the database
- **Modern UI**: Responsive design with glassmorphism effects

## Setup Instructions

### 1. Install Python Dependencies

```bash
# Run the setup script
python setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Add Known Faces

1. Create a `known_faces` folder
2. Add face images named as `PersonName.jpg` (e.g., `John_Doe.jpg`)
3. Or use the admin panel to add faces through the web interface

### 3. Start the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### 4. Open the Website

Open `index.html` in your browser or navigate to `http://localhost:5000`

## API Endpoints

- `POST /api/recognize` - Recognize faces in an image
- `POST /api/compare` - Compare two faces
- `POST /api/add_face` - Add a new face to database
- `GET /api/known_faces` - Get list of known faces

## Usage

1. **Upload Mode**: Drag & drop or select image files
2. **Camera Mode**: Use live webcam for real-time detection
3. **Compare Mode**: Upload two images to compare faces
4. **Admin Panel**: Add new people to the face database

## Requirements

- Python 3.7+
- OpenCV
- face_recognition library
- Flask
- Modern web browser with camera access

## File Structure

```
fcerecog/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── setup.py           # Setup script
├── index.html         # Landing page
├── login.html         # Authentication
├── dashboard.html     # User dashboard
├── recognition.html   # Face recognition interface
├── admin.html         # Admin panel
├── styles.css         # Main styles
├── recognition.css    # Recognition page styles
├── auth.css          # Authentication styles
├── dashboard.css     # Dashboard styles
├── script.js         # Main JavaScript
├── recognition.js    # Recognition functionality
├── auth.js           # Authentication logic
├── dashboard.js      # Dashboard functionality
├── known_faces/      # Known face images
├── uploads/          # Uploaded images
└── README.md         # This file
```

## Demo Accounts

- **Admin**: `admin@facerecog.com` / `admin123`
- **User**: `user@facerecog.com` / `user123`

## Notes

- Ensure good lighting for better face recognition accuracy
- Use clear, front-facing photos for known faces
- The system works best with high-quality images
- Camera access requires HTTPS in production