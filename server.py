import os
import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_file
import tempfile
import json
from werkzeug.utils import secure_filename
import cv2
from datetime import datetime
from flask_cors import CORS
from pyngrok import ngrok, conf

# Import your model components here
# from stable_audio_tools.inference.utils import prepare_audio
# from stable_audio_tools.models.transformer import Transformer

# Set up Ngrok with your auth token
ngrok.set_auth_token("2QagkaJp7QtxKlZBkke3zvjSZeC_71LtaJFng5mSZ2jKwb3SZ")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Helper function to save uploaded file
def save_uploaded_file(file, folder):
    if file:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(folder, new_filename)
        file.save(filepath)
        return filepath
    return None

# Helper function to generate a response with audio file
def generate_audio_response(audio_data, sample_rate=44100):
    # Create a temporary file to save the audio
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    
    # Save the audio data to the file
    torchaudio.save(output_path, audio_data, sample_rate)
    
    # Return the file
    return send_file(output_path, mimetype='audio/wav')

# Text-to-Audio (T2A) endpoint
@app.route('/t2a', methods=['POST'])
def text_to_audio():
    try:
        # Get text prompt from request
        data = request.get_json()
        if not data or 'text_prompt' not in data:
            return jsonify({"error": "Text prompt is required"}), 400
            
        text_prompt = data['text_prompt']
        
        # Here you would process the text prompt with your model
        # For demonstration, we'll create a simple sine wave
        sample_rate = 44100
        duration = 5  # seconds
        
        # Generate a simple audio signal (replace with your model's output)
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio_data = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
        
        # Return the generated audio
        return generate_audio_response(audio_data, sample_rate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Text-to-Music (T2M) endpoint
@app.route('/t2m', methods=['POST'])
def text_to_music():
    try:
        # Get text prompt from request
        data = request.get_json()
        if not data or 'text_prompt' not in data:
            return jsonify({"error": "Text prompt is required"}), 400
            
        text_prompt = data['text_prompt']
        
        # Here you would process the text prompt with your music generation model
        # For demonstration, we'll create a simple chord progression
        sample_rate = 44100
        duration = 5  # seconds
        
        # Generate a simple music signal (replace with your model's output)
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple chord progression (C major, G major, A minor, F major)
        frequencies = [
            [261.63, 329.63, 392.00],  # C major
            [392.00, 493.88, 587.33],  # G major
            [220.00, 277.18, 329.63],  # A minor
            [349.23, 440.00, 523.25]   # F major
        ]
        
        chord_duration = duration / len(frequencies)
        audio_data = torch.zeros(1, int(sample_rate * duration))
        
        for i, chord in enumerate(frequencies):
            start_idx = int(i * chord_duration * sample_rate)
            end_idx = int((i + 1) * chord_duration * sample_rate)
            t_chord = torch.linspace(0, chord_duration, end_idx - start_idx)
            
            chord_signal = torch.zeros(end_idx - start_idx)
            for freq in chord:
                chord_signal += torch.sin(2 * np.pi * freq * t_chord) / len(chord)
            
            audio_data[0, start_idx:end_idx] = chord_signal
        
        # Return the generated music
        return generate_audio_response(audio_data, sample_rate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Video-to-Audio (V2A) endpoint
@app.route('/v2a', methods=['POST'])
def video_to_audio():
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
            
        if not allowed_video_file(video_file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get text prompt if provided
        text_prompt = request.form.get('text_prompt', 'Generate general audio for the video')
        
        # Save the uploaded video
        video_path = save_uploaded_file(video_file, UPLOAD_FOLDER)
        
        # Here you would process the video with your model
        # For demonstration, we'll create a simple audio based on video duration
        
        # Get video duration using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 5  # Default to 5 seconds if can't determine
        cap.release()
        
        # Generate audio based on video duration
        sample_rate = 44100
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple audio signal that varies based on the video content
        # This is just a placeholder - your model would generate appropriate audio
        audio_data = torch.sin(2 * np.pi * 220 * t).unsqueeze(0)
        
        # Return the generated audio
        return generate_audio_response(audio_data, sample_rate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Video-to-Music (V2M) endpoint
@app.route('/v2m', methods=['POST'])
def video_to_music():
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
            
        if not allowed_video_file(video_file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get text prompt if provided
        text_prompt = request.form.get('text_prompt', 'Generate music for the video')
        
        # Save the uploaded video
        video_path = save_uploaded_file(video_file, UPLOAD_FOLDER)
        
        # Here you would process the video with your music generation model
        # For demonstration, we'll create a simple music based on video duration
        
        # Get video duration using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 5  # Default to 5 seconds if can't determine
        cap.release()
        
        # Generate music based on video duration
        sample_rate = 44100
        
        # Create a simple music signal
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a melody that changes over time
        audio_data = torch.zeros(1, int(sample_rate * duration))
        
        # Divide the duration into segments for different musical phrases
        num_segments = 4
        segment_duration = duration / num_segments
        
        for i in range(num_segments):
            start_idx = int(i * segment_duration * sample_rate)
            end_idx = int((i + 1) * segment_duration * sample_rate)
            t_segment = torch.linspace(0, segment_duration, end_idx - start_idx)
            
            # Different frequency for each segment
            freq = 220 * (1 + i * 0.25)  # Increasing pitch
            segment_signal = torch.sin(2 * np.pi * freq * t_segment)
            
            # Add some harmonics
            segment_signal += 0.5 * torch.sin(2 * np.pi * freq * 2 * t_segment)
            segment_signal += 0.25 * torch.sin(2 * np.pi * freq * 3 * t_segment)
            
            # Normalize
            segment_signal /= 1.75
            
            audio_data[0, start_idx:end_idx] = segment_signal
        
        # Return the generated music
        return generate_audio_response(audio_data, sample_rate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Text+Video-to-Audio (TV2A) endpoint
@app.route('/tv2a', methods=['POST'])
def tv_to_audio():
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
            
        if not allowed_video_file(video_file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get text prompt
        text_prompt = request.form.get('text_prompt')
        if not text_prompt:
            return jsonify({"error": "Text prompt is required"}), 400
        
        # Save the uploaded video
        video_path = save_uploaded_file(video_file, UPLOAD_FOLDER)
        
        # Here you would process the video and text with your model
        # For demonstration, we'll create a simple audio based on video duration and text
        
        # Get video duration using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 5  # Default to 5 seconds if can't determine
        cap.release()
        
        # Generate audio based on video duration and text prompt
        sample_rate = 44100
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple audio signal
        # In a real implementation, the text_prompt would influence the audio generation
        if "ocean" in text_prompt.lower():
            # Create ocean wave sounds (low frequency modulation)
            carrier = torch.sin(2 * np.pi * 100 * t)
            modulator = torch.sin(2 * np.pi * 0.1 * t)
            audio_data = (carrier * (1 + 0.5 * modulator)).unsqueeze(0)
        elif "laughing" in text_prompt.lower():
            # Create a sound with higher frequencies and variations
            audio_data = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
            audio_data += 0.5 * torch.sin(2 * np.pi * 880 * t).unsqueeze(0)
            # Add some random variations
            audio_data += 0.2 * torch.randn_like(audio_data)
        else:
            # Default audio
            audio_data = torch.sin(2 * np.pi * 330 * t).unsqueeze(0)
        
        # Return the generated audio
        return generate_audio_response(audio_data, sample_rate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Text+Video-to-Music (TV2M) endpoint
@app.route('/tv2m', methods=['POST'])
def tv_to_music():
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
            
        if not allowed_video_file(video_file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get text prompt
        text_prompt = request.form.get('text_prompt')
        if not text_prompt:
            return jsonify({"error": "Text prompt is required"}), 400
        
        # Save the uploaded video
        video_path = save_uploaded_file(video_file, UPLOAD_FOLDER)
        
        # Here you would process the video and text with your music generation model
        # For demonstration, we'll create a simple music based on video duration and text
        
        # Get video duration using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 5  # Default to 5 seconds if can't determine
        cap.release()
        
        # Generate music based on video duration and text prompt
        sample_rate = 44100
        
        # Create music based on the text prompt
        audio_data = torch.zeros(1, int(sample_rate * duration))
        
        if "piano" in text_prompt.lower():
            # Create piano-like sounds
            # Divide into measures
            measures = 8
            measure_duration = duration / measures
            notes_per_measure = 4
            note_duration = measure_duration / notes_per_measure
            
            # Piano frequencies (C major scale)
            piano_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
            
            for measure in range(measures):
                for note in range(notes_per_measure):
                    start_idx = int((measure * measure_duration + note * note_duration) * sample_rate)
                    end_idx = int((measure * measure_duration + (note + 1) * note_duration) * sample_rate)
                    
                    # Select a note from the scale
                    note_idx = (measure + note) % len(piano_notes)
                    freq = piano_notes[note_idx]
                    
                    # Create the note
                    t_note = torch.linspace(0, note_duration, end_idx - start_idx)
                    
                    # Piano-like envelope (attack, decay, sustain, release)
                    attack = int(0.05 * (end_idx - start_idx))
                    decay = int(0.1 * (end_idx - start_idx))
                    release = int(0.2 * (end_idx - start_idx))
                    
                    envelope = torch.ones(end_idx - start_idx)
                    # Attack
                    envelope[:attack] = torch.linspace(0, 1, attack)
                    # Decay
                    envelope[attack:attack+decay] = torch.linspace(1, 0.7, decay)
                    # Release
                    envelope[-release:] = torch.linspace(0.7, 0, release)
                    
                    # Apply envelope to a sine wave
                    note_signal = torch.sin(2 * np.pi * freq * t_note) * envelope
                    
                    # Add some harmonics for richness
                    note_signal += 0.5 * torch.sin(2 * np.pi * freq * 2 * t_note) * envelope
                    note_signal += 0.25 * torch.sin(2 * np.pi * freq * 3 * t_note) * envelope
                    
                    # Normalize
                    note_signal /= 1.75
                    
                    audio_data[0, start_idx:end_idx] = note_signal
        else:
            # Default music
            t = torch.linspace(0, duration, int(sample_rate * duration))
            audio_data = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        # Return the generated music
        return generate_audio_response(audio_data, sample_rate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "MCP Audio Server"})

# Root endpoint to show available endpoints
@app.route('/', methods=['GET'])
def index():
    endpoints = {
        "endpoints": [
            {"path": "/t2a", "method": "POST", "description": "Text-to-Audio generation"},
            {"path": "/t2m", "method": "POST", "description": "Text-to-Music generation"},
            {"path": "/v2a", "method": "POST", "description": "Video-to-Audio generation"},
            {"path": "/v2m", "method": "POST", "description": "Video-to-Music generation"},
            {"path": "/tv2a", "method": "POST", "description": "Text+Video-to-Audio generation"},
            {"path": "/tv2m", "method": "POST", "description": "Text+Video-to-Music generation"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ],
        "service": "MCP Audio Server"
    }
    return jsonify(endpoints)

if __name__ == '__main__':
    # Set up ngrok tunnel
    port = 5000
    
    # Connect to ngrok with your reserved domain
    try:
        # For a static/reserved domain, use domain instead of subdomain
        public_url = ngrok.connect(port, domain="mint-gator-truly.ngrok-free.app").public_url
    except Exception as e:
        # If that fails, use a random URL
        print(f"Could not use reserved domain: {str(e)}")
        public_url = ngrok.connect(port).public_url
    
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    
    # Update any base URLs or callback URLs with the public URL
    app.config['BASE_URL'] = public_url
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=port, debug=True)