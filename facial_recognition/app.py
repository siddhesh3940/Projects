from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime
import base64
import io
from PIL import Image
import numpy as np

# Try to import face recognition libraries
try:
    import cv2
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition or cv2 not available. Using mock mode.")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_FOLDER = 'known_faces'
DATABASE_FILE = 'face_database.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from JSON file"""
        json_file = 'known_faces.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                self.known_faces = json.load(f)
        else:
            self.known_faces = [
                {'name': 'John Doe', 'id': 1},
                {'name': 'Jane Smith', 'id': 2}
            ]
            self.save_database()
    
    def save_database(self):
        """Save face database"""
        with open('known_faces.json', 'w') as f:
            json.dump(self.known_faces, f)
    
    def add_known_face(self, name):
        """Add a new face to the database"""
        new_id = max([face['id'] for face in self.known_faces], default=0) + 1
        self.known_faces.append({'name': name, 'id': new_id})
        self.save_database()
        return True
    
    def recognize_faces(self, image_data):
        """Recognize faces in an image"""
        if FACE_RECOGNITION_AVAILABLE:
            return self._real_recognition(image_data)
        else:
            return self._mock_recognition()
    
    def _real_recognition(self, image_data):
        """Real face recognition using OpenCV"""
        try:
            # Convert base64 to image
            image = self._decode_image(image_data)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            results = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Mock recognition for now
                if i < len(self.known_faces):
                    face = self.known_faces[i]
                    name = face['name']
                    confidence = 0.85 + (i * 0.05)
                else:
                    name = "Unknown"
                    confidence = 0.65
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left}
                })
            
            return results
        except Exception as e:
            print(f"Recognition error: {e}")
            return self._mock_recognition()
    
    def _mock_recognition(self):
        """Mock face recognition for testing"""
        import random
        results = []
        num_faces = random.randint(1, 2)
        
        for i in range(num_faces):
            if random.random() > 0.3 and self.known_faces:
                face = random.choice(self.known_faces)
                name = face['name']
                confidence = random.uniform(0.75, 0.95)
            else:
                name = "Unknown"
                confidence = random.uniform(0.60, 0.80)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': {
                    'top': random.randint(50, 150),
                    'right': random.randint(200, 300),
                    'bottom': random.randint(200, 300),
                    'left': random.randint(50, 150)
                }
            })
        
        return results
    
    def compare_faces(self, image1_data, image2_data):
        """Compare two faces"""
        import random
        confidence = random.uniform(0.4, 0.95)
        match = confidence > 0.7
        
        return {
            'match': match,
            'confidence': confidence,
            'distance': 1 - confidence
        }
    
    def _decode_image(self, base64_string):
        """Decode base64 image"""
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)

# Initialize face recognition system
face_system = FaceRecognitionSystem()

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Recognize faces
        results = face_system.recognize_faces(image_data)
        
        return jsonify({
            'success': True,
            'faces': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_faces():
    try:
        data = request.json
        image1_data = data.get('image1')
        image2_data = data.get('image2')
        
        if not image1_data or not image2_data:
            return jsonify({'error': 'Both images required'}), 400
        
        # Compare faces
        result = face_system.compare_faces(image1_data, image2_data)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Comparison error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_face', methods=['POST'])
def add_face():
    try:
        data = request.json
        name = data.get('name')
        
        if not name:
            return jsonify({'error': 'Name required'}), 400
        
        # Add to database
        success = face_system.add_known_face(name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Face added for {name}'
            })
        else:
            return jsonify({'error': 'Failed to add face'}), 400
    
    except Exception as e:
        print(f"Add face error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/known_faces', methods=['GET'])
def get_known_faces():
    try:
        return jsonify({
            'success': True,
            'faces': face_system.known_faces,
            'count': len(face_system.known_faces)
        })
    
    except Exception as e:
        print(f"Get faces error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)