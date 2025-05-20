import os
import cv2
import time
import pickle
import json
import numpy as np
import queue
import sounddevice as sd
import vosk
import face_recognition
from collections import Counter
import tensorflow.lite as tflite  # Changed from full tensorflow
from ultralytics import YOLO
from gtts import gTTS
import pygame
from pygame import mixer

# ========================
# Performance Optimizations
# ========================
# Reduce logging and set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

MODELS = {
    'face': None,
    'currency': None,
    'object': None,
    'vosk': None
}

def load_models():
    """Pre-load all models at startup"""
    print("[SYSTEM] Loading models...")

    # VOSK model - use absolute path
    vosk_model_path = os.path.expanduser("/home/yasser/my_sight_project/Models/vosk-model-small-en-us-0.15")
    if os.path.exists(vosk_model_path):
        MODELS['vosk'] = vosk.Model(vosk_model_path)

    # Face recognition - use absolute path
    face_model_path = os.path.expanduser("/home/yasser/my_sight_project/Models/face_recognition_model.pkl")
    if os.path.exists(face_model_path):
        with open(face_model_path, "rb") as f:
            MODELS['face'] = pickle.load(f)

    # Currency detection - use absolute path and forward slashes
    currency_model_path = os.path.expanduser("/home/yasser/my_sight_project/Models/currency_int8 (1).tflite")
    currency_index_path = os.path.expanduser("/home/yasser/my_sight_project/Models/class_indices.json")
    if all(os.path.exists(p) for p in [currency_model_path, currency_index_path]):
        interpreter = tflite.Interpreter(model_path=currency_model_path)
        interpreter.allocate_tensors()
        MODELS['currency'] = {
            'interpreter': interpreter,
            'class_map': json.load(open(currency_index_path, "r"))
        }

    # Object detection - use absolute path
    object_model_path = os.path.expanduser("/home/yasser/my_sight_project/Models/yolov8n.pt")
    if os.path.exists(object_model_path):
        MODELS['object'] = {'model': YOLO(object_model_path)}

# Initialize pygame with reduced buffer
pygame.init()
mixer.init(buffer=512)  # Smaller buffer for lower latency

# ========================
# Optimized TTS Function
# ========================
def speak(text):
    """Optimized text-to-speech function for Raspberry Pi"""
    try:
        temp_file = f"/tmp/tts_{int(time.time())}.mp3"  # Use /tmp which is in RAM
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(temp_file)
        
        mixer.music.load(temp_file)
        mixer.music.play()
        
        # Calculate duration based on word count
        duration = min(max(len(text.split()) * 0.3, 1), 3)  # 0.3s per word, 1-3s max
        pygame.time.wait(int(duration * 1000))
        
        # Cleanup
        try:
            os.remove(temp_file)
        except:
            pass
    except Exception as e:
        print(f"TTS Error: {str(e)[:100]}")  # Truncate long errors

# ========================
# Optimized Voice Command
# ========================
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata)) if not status else None

def listen_command(timeout=5):
    """Voice command recognition optimized for Pi"""
    if not MODELS['vosk']:
        return None
        
    with sd.RawInputStream(samplerate=16000, blocksize=8000,  # Increased blocksize for Pi
                          dtype='int16', channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(MODELS['vosk'], 16000)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                command = result.get("text", "").strip().lower()
                if command in ["face", "currency", "color", "object", "exit", "quit"]:
                    return command
        return None

# ========================
# Optimized Feature Functions
# ========================
def recognize_face():
    """Optimized face recognition for Raspberry Pi"""
    try:
        if not MODELS['face']:
            speak("Face model not loaded")
            return

        # Camera setup with longer warmup
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        time.sleep(2)  # Camera warmup
        
        # Capture multiple frames to get a good one
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                break
        cap.release()
        
        if not ret:
            speak("Camera error")
            return

        # Debug: Save captured image
        cv2.imwrite("debug_face.jpg", frame)
        print("Debug image saved as debug_face.jpg")

        # Convert to RGB and resize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

        # Try both detection methods
        face_locations = face_recognition.face_locations(small_frame, model="hog")  # Faster
        if not face_locations:
            face_locations = face_recognition.face_locations(small_frame, model="cnn")  # More accurate
            
        if not face_locations:
            speak("No faces detected. Please ensure good lighting.")
            return

        # Process all found faces
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(MODELS['face']["encodings"], face_encoding, tolerance=0.5)
            
            if True in matches:
                matched_names = [MODELS['face']["names"][i] for i, match in enumerate(matches) if match]
                name = max(set(matched_names), key=matched_names.count)
                speak(f"{name}")
            else:
                speak("Unknown person detected")

    except Exception as e:
        print(f"Face recognition error: {str(e)[:100]}")
        speak("Face recognition service unavailable")

def recognize_currency():
    """Currency detection optimized for Pi"""
    try:
        if not MODELS['currency']:
            speak("Currency model not loaded")
            return

        # Reduced resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            speak("Camera error")
            return

        # Get model components
        interpreter = MODELS['currency']['interpreter']
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Preprocess image
        img = cv2.resize(frame, (224, 224))
        img = img[:, :, ::-1]  # BGR to RGB
        
        # Handle quantization
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            img = (img / input_scale + input_zero_point).astype(np.uint8)
        else:
            img = (img / 127.5 - 1.0).astype(np.float32)  # Simple scaling

        # Inference
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Handle output quantization
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        # Get prediction
        predicted_idx = np.argmax(output[0])
        predicted_label = MODELS['currency']['class_map'][str(predicted_idx)]
        speak(f"Detected: {predicted_label}")
        
    except Exception as e:
        print(f"Currency Error: {str(e)[:100]}")
        speak("Currency detection failed")

def detect_color():
    """Color detection optimized for Pi"""
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            speak("Camera error")
            return

        # Smaller detection area
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        region = hsv[h//3:h*2//3, w//3:w*2//3]  # Center region
        
        # Dominant color
        pixels = region.reshape(-1, 3)
        dominant_color = np.median(pixels, axis=0)
        h, s, v = dominant_color
        
        # Simple color mapping
        if v < 40: color = "Black"
        elif s < 50: color = "White" if v > 200 else "Gray"
        elif h < 15: color = "Red"
        elif h < 30: color = "Orange"
        elif h < 45: color = "Yellow"
        elif h < 85: color = "Green"
        elif h < 130: color = "Blue"
        elif h < 165: color = "Purple"
        else: color = "Pink"
        
        speak(f"Color: {color}")
        
    except Exception as e:
        print(f"Color Error: {str(e)[:100]}")
        speak("Color detection failed")

def detect_object():
    """Object detection optimized for Pi"""
    try:
        if not MODELS['object']:
            speak("Object model not loaded")
            return

        # Reduced resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            speak("Camera error")
            return

        # Half-precision inference
        results = MODELS['object']['model'](frame, imgsz=320, half=True, verbose=False)[0]
        
        if len(results.boxes.cls) == 0:
            speak("No objects found")
            return

        # Most common object
        most_common = Counter(results.boxes.cls.cpu().numpy().astype(int)).most_common(1)[0][0]
        label = MODELS['object']['model'].names[most_common]
        speak(f"I see a {label}")
            
    except Exception as e:
        print(f"Object Error: {str(e)[:100]}")
        speak("Object detection failed")

# ========================
# Main Loop
# ========================
if __name__ == "__main__":
    # Pre-load models
    load_models()
    
    # Camera warmup
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
    
    speak("welcom to My Sight. Say a command: face, currency, color, object, or exit.")
    
    running = True
    while running:
        try:
            command = listen_command()
            
            if command == "face":
                recognize_face()
            elif command == "currency":
                recognize_currency()
            elif command == "color":
                detect_color()
            elif command == "object":
                detect_object()
            elif command in ["exit", "quit"]:
                speak("Goodbye")
                running = False
                
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"System Error: {str(e)[:100]}")
            time.sleep(1)
    
    pygame.quit()