import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
from datetime import datetime
import joblib

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    st.warning("DeepFace not available. Install with: pip install deepface")

st.set_page_config(page_title="Student Engagement System - AI Live Model", layout="wide")

# Database setup
def init_database():
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Engagement sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS engagement_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engagement_score REAL,
            emotion TEXT,
            confidence REAL,
            face_detected BOOLEAN,
            ai_prediction TEXT,
            ai_confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # Add AI columns if they don't exist
    try:
        cursor.execute("ALTER TABLE engagement_sessions ADD COLUMN ai_prediction TEXT")
        cursor.execute("ALTER TABLE engagement_sessions ADD COLUMN ai_confidence REAL")
    except sqlite3.OperationalError:
        pass  # Columns already exist
    
    conn.commit()
    conn.close()

init_database()
class MultiFacePersistentTracker:
    def __init__(self):
        self.face_memory = {}
        self.persistence_seconds = 20

    def get_face_key(self, x, y, w, h):
        # Bins coordinates to reduce twitching
        cx, cy = (x + w // 2) // 100 * 100, (y + h // 2) // 100 * 100
        return (cx, cy)

    def update(self, face_coords, student_info, confidence):
        """
        Track students by face location. Returns persistent identity.
        """
        key = self.get_face_key(*face_coords)
        now = time.time()
        entry = self.face_memory.get(key)

        if student_info and confidence > 0.1:
            # New or confirmed identification
            if not entry or entry['student_info'] != student_info:
                # Print only if new or changed
                print(f"🆕 Identified: {student_info[1]} at {key} (conf: {confidence:.3f})")
            self.face_memory[key] = {
                'student_info': student_info,
                'confidence': confidence,
                'last_seen': now,
            }
            return student_info, confidence

        # No match: check memory
        if entry and now - entry['last_seen'] < self.persistence_seconds:
            # Print only once on fallback
            if 'printed_unknown' not in entry:
                print(f"🔒 Keeping {entry['student_info'][1]} at {key}")
                self.face_memory[key]['printed_unknown'] = True
            return entry['student_info'], entry['confidence']
        else:
            # If timeout, remove memory for this face
            if key in self.face_memory:
                print(f"🗑️ Removing memory for face {key}")
                del self.face_memory[key]
            return None, 0.0

def delete_student(student_id):
    """Delete a student from the database"""
    try:
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        
        # First check if student exists
        cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
        student = cursor.fetchone()
        
        if student:
            student_name = student[0]
            
            # Delete from students table
            cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
            
            # Delete from engagement_sessions table
            cursor.execute('DELETE FROM engagement_sessions WHERE student_id = ?', (student_id,))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Deleted student: {student_name}")
            return True, f"Successfully deleted {student_name}"
        else:
            conn.close()
            return False, "Student not found"
            
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return False, f"Delete failed: {str(e)}"

def cleanup_invalid_students():
    """Remove students with invalid face encodings"""
    try:
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, student_id, name, face_encoding FROM students')
        students = cursor.fetchall()
        
        removed_count = 0
        for pk_id, student_id, name, encoding_str in students:
            try:
                if encoding_str is None or encoding_str == '':
                    cursor.execute('DELETE FROM students WHERE id = ?', (pk_id,))
                    removed_count += 1
                    continue
                
                # Test encoding validity
                test_encoding = eval(encoding_str)
                if not isinstance(test_encoding, (list, np.ndarray)) or len(test_encoding) == 0:
                    cursor.execute('DELETE FROM students WHERE id = ?', (pk_id,))
                    removed_count += 1
                    
            except Exception:
                cursor.execute('DELETE FROM students WHERE id = ?', (pk_id,))
                removed_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Cleanup complete. Removed {removed_count} invalid students.")
        return True, f"Removed {removed_count} invalid entries"
        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return False, str(e)

# Enhanced Model Loader Class
class TrainedModelLoader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False
        self.model_info = "No model"
        self.load_trained_model()
    
    def load_trained_model(self):
        try:
            import glob
            model_files = glob.glob('best_engagement_model_*.pkl')
            scaler_files = glob.glob('scaler_*.pkl')
            encoder_files = glob.glob('label_encoder_*.pkl')
            
            if model_files and scaler_files:
                latest_model = sorted(model_files)[-1]
                latest_scaler = sorted(scaler_files)[-1]
                
                self.model = joblib.load(latest_model)
                self.scaler = joblib.load(latest_scaler)
                
                if encoder_files:
                    self.label_encoder = joblib.load(sorted(encoder_files)[-1])
                else:
                    # Create manual mapping if encoder missing
                    class SimpleEncoder:
                        def __init__(self):
                            self.classes_ = ['engaged', 'neutral', 'not_engaged']
                        def inverse_transform(self, x):
                            mapping = {0: 'engaged', 1: 'neutral', 2: 'not_engaged'}
                            return [mapping.get(i, 'unknown') for i in x]
                    self.label_encoder = SimpleEncoder()
                
                self.model_loaded = True
                self.model_info = f"✅ AI Model Ready"
                print(f"✅ AI Model loaded: {os.path.basename(latest_model)}")
                
            else:
                print("❌ Model files missing!")
                print(f"Models found: {len(model_files)}")
                print(f"Scalers found: {len(scaler_files)}")
                self.model_info = "❌ No model files found"
                
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            self.model_info = f"❌ Error: {str(e)[:30]}"
    
    def predict_engagement(self, engagement_score, emotion, confidence, face_detected):
        if not self.model_loaded:
        # fallback logic unchanged
            if engagement_score >= 0.6:
                return "engaged", 0.75
            elif engagement_score <= 0.3:
                return "not_engaged", 0.75
            else:
                return "neutral", 0.65

        try:
            emotion_map = {
                'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3,
                'surprise': 4, 'disgust': 5, 'neutral': 6
            }
            emotion_encoded = emotion_map.get(emotion.lower(), 6)

            # HERE IS THE FIX: Provide the same 7 features as for training
            feature_1 = float(engagement_score)
            feature_2 = float(emotion_encoded)
            feature_3 = float(confidence)
            feature_4 = float(int(face_detected))

            feature_5 = feature_1 * feature_3  # engagement_confidence
            feature_6 = feature_1 ** 2         # engagement_squared
            feature_7 = feature_3 ** 2         # confidence_squared

            features = np.array([[
            feature_1,
            feature_2,
            feature_3,
            feature_4,
            feature_5,
            feature_6,
            feature_7
            ]])

            features_scaled = self.scaler.transform(features)
            prediction = int(self.model.predict(features_scaled)[0])

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                conf = float(max(probabilities))
            else:
                conf = 0.80

            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            return predicted_label, conf

        except Exception as e:
            print(f"⚠️ AI prediction fallback due to: {e}")
            if engagement_score >= 0.6:
                return "engaged", 0.70
            elif engagement_score <= 0.3:
                return "not_engaged", 0.70
            else:
                return "neutral", 0.60

# FIXED: Video Processing Class that persists in session state
class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.running = False
        
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                self.running = True
                return True
            else:
                self.cap = None
                return False
        return self.running
        
    def get_frame(self):
        if self.cap and self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
        
    def stop_camera(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        
    def is_running(self):
        return self.running and self.cap and self.cap.isOpened()

# FIXED: Initialize components in session state to persist across reruns
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = TrainedModelLoader()

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()

# Initialize session state
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_frames": 0,
        "engaged_frames": 0,
        "students_detected": set(),
        "current_session": [],
        "live_mode": False
    }

if 'video_running' not in st.session_state:
    st.session_state.video_running = False

if 'student_tracker' not in st.session_state:
    st.session_state.student_tracker = MultiFacePersistentTracker()


if 'last_frame_time' not in st.session_state:
    st.session_state.last_frame_time = 0

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade

face_cascade = load_models()

# Face recognition functions (same as before)
def register_student(name, student_id, image):
    """Enhanced student registration with consistent face encoding"""
    if not DEEPFACE_AVAILABLE:
        return False, "DeepFace not available"
    
    try:
        # Ensure consistent image size and quality
        if image.shape[0] < 150 or image.shape[1] < 150:
            image = cv2.resize(image, (160, 160))
        
        # Apply same enhancement as identification
        image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        
        temp_path = f"temp_{student_id}.jpg"
        cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Get face embedding with same settings as identification
        try:
            result = DeepFace.represent(
                img_path=temp_path, 
                model_name="VGG-Face",
                enforce_detection=False
            )
            embedding = result[0]['embedding'] if isinstance(result, list) else result['embedding']
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, f"Face encoding failed: {str(e)}"
        
        # Store in database
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO students (name, student_id, face_encoding)
            VALUES (?, ?, ?)
        ''', (name, student_id, str(embedding)))
        
        conn.commit()
        conn.close()
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"✅ Registered {name} successfully")
        return True, f"Student {name} registered successfully"
        
    except Exception as e:
        if os.path.exists(f"temp_{student_id}.jpg"):
            os.remove(f"temp_{student_id}.jpg")
        return False, f"Registration failed: {str(e)}"

def identify_student(image):
    """Fixed with higher threshold"""
    if not DEEPFACE_AVAILABLE:
        return None, 0.0
    
    try:
        if image.shape[0] < 120 or image.shape[1] < 120:
            image = cv2.resize(image, (160, 160))
        
        image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        temp_path = "temp_identify.jpg"
        cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Get students
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        cursor.execute('SELECT student_id, name, face_encoding FROM students')
        students = cursor.fetchall()
        conn.close()
        
        if not students:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None, 0.0
        
        print(f"🔍 Checking against {len(students)} students")
        
        # Get embedding
        try:
            result = DeepFace.represent(temp_path, model_name="VGG-Face", enforce_detection=False)
            current_embedding = result[0]['embedding'] if isinstance(result, list) else result['embedding']
        except Exception as e:
            print(f"⚠️ Face embedding failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None, 0.0
        
        # Find best match
        best_match = None
        best_distance = float('inf')
        
        for student_id, name, encoding_str in students:
            try:
                if not encoding_str:
                    continue
                
                stored_embedding = eval(encoding_str)
                distance = np.linalg.norm(np.array(current_embedding) - np.array(stored_embedding))
                print(f"🎯 {name}: distance = {distance:.3f}")
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = (student_id, name)
                    
            except Exception as e:
                print(f"⚠️ Error matching {name}: {e}")
                continue
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # FIXED: Higher threshold - your 0.964 will now match!
        if best_match and best_distance < 1.2:
            confidence = max(0.1, 1 - (best_distance / 1.2))
            print(f"✅ Identified: {best_match[1]} (distance: {best_distance:.3f}, confidence: {confidence:.3f})")
            return best_match, confidence
        else:
            print(f"❌ No match - best distance: {best_distance:.3f} (threshold: 1.2)")
        
        return None, 0.0
        
    except Exception as e:
        print(f"❌ Identification error: {e}")
        if os.path.exists("temp_identify.jpg"):
            os.remove("temp_identify.jpg")
        return None, 0.0

def detect_emotion_engagement(image):
    """Detect emotion and compute engagement score"""
    if not DEEPFACE_AVAILABLE:
        return "neutral", 0.5
    
    try:
        temp_path = "temp_emotion.jpg"
        cv2.imwrite(temp_path, image)
        
        result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        
        engagement_score = 0.0
        engagement_score += emotions.get('happy', 0) * 0.8
        engagement_score += emotions.get('surprise', 0) * 0.6
        engagement_score += emotions.get('neutral', 0) * 0.5
        engagement_score += emotions.get('sad', 0) * 0.2
        engagement_score += emotions.get('angry', 0) * 0.1
        engagement_score += emotions.get('fear', 0) * 0.1
        engagement_score += emotions.get('disgust', 0) * 0.1
        
        engagement_score = engagement_score / 100.0
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return dominant_emotion, engagement_score
        
    except Exception as e:
        if os.path.exists("temp_emotion.jpg"):
            os.remove("temp_emotion.jpg")
        return "neutral", 0.5

def log_engagement_data_with_ai(student_id, engagement_score, emotion, confidence, face_detected, ai_prediction="", ai_confidence=0.0):
    """Enhanced logging with AI predictions"""
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO engagement_sessions 
            (student_id, engagement_score, emotion, confidence, face_detected, ai_prediction, ai_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (student_id, engagement_score, emotion, confidence, face_detected, ai_prediction, ai_confidence))
    except sqlite3.OperationalError:
        cursor.execute('''
            INSERT INTO engagement_sessions 
            (student_id, engagement_score, emotion, confidence, face_detected)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_id, engagement_score, emotion, confidence, face_detected))
    
    conn.commit()
    conn.close()

# UI Layout
st.title("🎓 AI-Powered Student Engagement Detection System")
st.caption("Real-time face recognition and engagement monitoring with trained AI model")

# Sidebar controls
with st.sidebar:
    st.header("🔧 Controls")
    
    mode = st.selectbox("Mode", ["Live Detection", "Student Registration", "Analytics Dashboard"])
    
    # AI Model Status
    st.markdown("---")
    st.subheader("🤖 AI Model Status")
    
    if st.session_state.trained_model.model_loaded:
        st.success(st.session_state.trained_model.model_info)
        st.info("🧠 AI predictions active")
    else:
        st.error(st.session_state.trained_model.model_info)
        st.warning("Using basic detection")
    
    if st.button("🔄 Reload AI Model"):
        st.session_state.trained_model.load_trained_model()
        st.rerun()
    
    if mode == "Live Detection":
        st.markdown("---")
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Recognition Confidence Threshold", 0.1, 1.0, 0.6)
        engagement_threshold = st.slider("Engagement Threshold", 0.1, 1.0, 0.5)
        
        # Video controls
        st.markdown("---")
        st.subheader("📹 Video Controls")
        is_running = st.session_state.video_processor.is_running()
        st.write(f"Camera Status: {'🔴 LIVE' if is_running else '⚪ Stopped'}")
    
    elif mode == "Student Registration":
        st.subheader("Register New Student")
        new_name = st.text_input("Student Name")
        new_id = st.text_input("Student ID")
        
    st.markdown("---")
    if st.button("Reset Session Data"):
        st.session_state.stats = {
            "total_frames": 0,
            "engaged_frames": 0,
            "students_detected": set(),
            "current_session": [],
            "live_mode": True
        }
        st.success("Session data reset!")

# Main content area
if mode == "Live Detection":
    st.subheader("📹 AI-Powered Live Real-time Detection")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("▶️ Start Live Stream"):
            if st.session_state.video_processor.start_camera():
                st.session_state.video_running = True
                st.success("🟢 Camera started!")
                st.rerun()
            else:
                st.error("❌ Camera not available")
                
    with col2:
        if st.button("⏹️ Stop Stream"):
            st.session_state.video_processor.stop_camera()
            st.session_state.video_running = False
            st.info("⏹️ Camera stopped")
            st.rerun()
            
    with col3:
        if st.button("📸 Capture & Log"):
            st.session_state.capture_frame = True
            
    with col4:
        is_running = st.session_state.video_processor.is_running()
        st.write(f"**Status:** {'🔴 LIVE' if is_running else '⚪ Stopped'}")
    
    # FIXED: Continuous Live processing without aggressive reruns
    if st.session_state.video_processor.is_running():
        
        # Create containers that will be updated
        frame_container = st.container()
        stats_container = st.container()
        
        # Process frames continuously
        current_time = time.time()
        
        # Get frame from video processor
        frame = st.session_state.video_processor.get_frame()
        
        if frame is not None:
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            annotated_frame = frame.copy()
            current_detections = []
            
            # Process each detected face
                        # Process each detected face
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                
                # UPDATED: Get student info with persistence
                raw_student_info, raw_confidence = identify_student(face_region)
                face_key = (x, y, w, h)
                student_info, recognition_confidence = st.session_state.student_tracker.update(
                    (x, y, w, h), raw_student_info, raw_confidence
                    )
                
                # Get emotion and basic engagement
                emotion, engagement_score = detect_emotion_engagement(face_region)

                
                # Get AI prediction
                ai_prediction, ai_confidence = st.session_state.trained_model.predict_engagement(
                    engagement_score, emotion, recognition_confidence, True
                )
                
                # Update statistics
                st.session_state.stats["total_frames"] += 1
                if ai_prediction == "engaged":
                    st.session_state.stats["engaged_frames"] += 1
                
                                # FIXED: Use persistent tracker results for display
                if student_info and recognition_confidence >= 0.15:  # Very low threshold
                    student_id, student_name = student_info
                    st.session_state.stats["students_detected"].add(student_name)
                    print(f"📱 UI: Showing {student_name} (conf: {recognition_confidence:.3f})")
                else:
                    student_id, student_name = "unknown", "Unknown Student"
                    print(f"📱 UI: Showing Unknown Student")

                # Log data if capture requested
                if hasattr(st.session_state, 'capture_frame') and st.session_state.capture_frame:
                    log_engagement_data_with_ai(student_id, engagement_score, emotion, 
                                              recognition_confidence, True, ai_prediction, ai_confidence)
                
                # Store detection
                current_detections.append({
                    'name': student_name,
                    'emotion': emotion,
                    'engagement_score': engagement_score,
                    'ai_prediction': ai_prediction,
                    'ai_confidence': ai_confidence,
                    'recognition_confidence': recognition_confidence
                })
                
                # Draw annotations with AI prediction colors
                if ai_prediction == "engaged":
                    color = (0, 255, 0)  # Green
                elif ai_prediction == "not_engaged":
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 165, 0)  # Orange
                
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 3)
                
                # Labels
                cv2.putText(annotated_frame, f"{student_name} ({recognition_confidence:.2f})", 
                           (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.putText(annotated_frame, f"{emotion.title()} - Score: {engagement_score:.2f}", 
                           (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(annotated_frame, f"AI: {ai_prediction.upper()} ({ai_confidence:.2f})", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Reset capture flag
            if hasattr(st.session_state, 'capture_frame'):
                st.session_state.capture_frame = False
            
            # Display the annotated frame
            with frame_container:
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                        caption="🔴 LIVE - AI Engagement Detection", 
                        use_column_width=True)
            
            # Display current stats
            with stats_container:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if current_detections:
                        st.write("### 📊 Current Live Detections:")
                        for i, detection in enumerate(current_detections):
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("Student", detection['name'][:15])
                            with col_b:
                                st.metric("Emotion", detection['emotion'].title())
                            with col_c:
                                st.metric("Basic Score", f"{detection['engagement_score']:.2f}")
                            with col_d:
                                pred = detection['ai_prediction']
                                emoji = "🟢" if pred == "engaged" else "🔴" if pred == "not_engaged" else "🟡"
                                st.metric("AI Prediction", f"{emoji} {pred.upper()}")
                    else:
                        st.info("👤 No faces detected in current frame")
                
                with col2:
                    st.write("### 📈 Session Stats")
                    total_frames = st.session_state.stats["total_frames"]
                    engaged_frames = st.session_state.stats["engaged_frames"]
                    engagement_rate = (engaged_frames / total_frames * 100) if total_frames > 0 else 0
                    
                    st.metric("Total Frames", total_frames)
                    st.metric("AI Engagement Rate", f"{engagement_rate:.1f}%")
                    st.metric("Students Detected", len(st.session_state.stats["students_detected"]))
                    
                    if st.session_state.stats["students_detected"]:
                        st.write("**Detected Students:**")
                        for student in list(st.session_state.stats["students_detected"])[:5]:
                            st.write(f"• {student}")
            
            # FIXED: Use time-based refresh instead of immediate rerun
            st.session_state.last_frame_time = current_time
            time.sleep(0.1)  # Small delay to prevent overwhelming
            st.rerun()
        
        else:
            st.warning("📷 Camera feed interrupted. Trying to reconnect...")
            # Try to restart camera
            if st.session_state.video_processor.start_camera():
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Camera connection lost. Click 'Start Live Stream' to restart.")
    
    else:
        st.info("📹 Click '▶️ Start Live Stream' to begin real-time AI engagement detection")
        st.markdown("""
        ### 🚀 Features:
        - **Real-time AI predictions** using your trained model
        - **Color-coded detection**: Green (Engaged), Red (Not Engaged), Orange (Neutral)  
        - **Live statistics** and engagement tracking
        - **Student recognition** with confidence scores
        - **Automatic logging** when you click 'Capture & Log'
        - **Continuous streaming** until you click 'Stop Stream'
        """)

elif mode == "Student Registration":
    st.subheader("👤 Student Management")
    
    # Create tabs for Register and Manage
    tab1, tab2 = st.tabs(["📝 Register New Student", "🗑️ Manage Students"])
    
    with tab1:
        # Original registration functionality
        col1, col2 = st.columns([1, 1])
        
        with col1:
            registration_img = st.camera_input("Capture student photo for registration")
            
            if registration_img is not None and new_name and new_id:
                if st.button("Register Student"):
                    pil_image = Image.open(registration_img)
                    rgb_image = np.array(pil_image)
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    
                    success, message = register_student(new_name, new_id, bgr_image)
                    
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        with col2:
            st.subheader("📋 Registered Students")
            
            conn = sqlite3.connect('student_engagement.db')
            students_df = pd.read_sql_query('SELECT student_id, name, created_at FROM students ORDER BY created_at DESC', conn)
            conn.close()
            
            if not students_df.empty:
                st.dataframe(students_df, use_container_width=True)
            else:
                st.info("No students registered yet.")
    
    with tab2:
        # NEW: Student management functionality
        st.subheader("🗑️ Delete Students")
        
        # Get all students for selection
        conn = sqlite3.connect('student_engagement.db')
        try:
            students_df = pd.read_sql_query('SELECT student_id, name FROM students ORDER BY name', conn)
        except:
            students_df = pd.DataFrame()
        conn.close()
        
        if not students_df.empty:
            # Select student to delete
            student_options = [f"{row['name']} (ID: {row['student_id']})" for _, row in students_df.iterrows()]
            selected_student = st.selectbox("Select student to delete:", [""] + student_options)
            
            if selected_student:
                student_id = selected_student.split("ID: ")[1].rstrip(")")
                student_name = selected_student.split(" (ID:")[0]
                
                st.warning(f"⚠️ You are about to delete: **{student_name}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Confirm Delete", type="secondary"):
                        success, message = delete_student(student_id)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                with col2:
                    if st.button("🧹 Cleanup Invalid Data"):
                        success, message = cleanup_invalid_students()
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            # Show cleanup button even if no student selected
            st.markdown("---")
            if st.button("🔧 Clean All Invalid Encodings"):
                success, message = cleanup_invalid_students()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.info("No students to manage.")
            
            # Show cleanup option even with no students
            if st.button("🔧 Clean Database"):
                success, message = cleanup_invalid_students()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

elif mode == "Analytics Dashboard":
    st.subheader("📈 AI Analytics Dashboard")
    
    # Load engagement data
    conn = sqlite3.connect('student_engagement.db')
    
    # Overall statistics
    engagement_df = pd.read_sql_query('''
        SELECT 
            DATE(timestamp) as date,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as total_detections,
            SUM(CASE WHEN face_detected THEN 1 ELSE 0 END) as faces_detected
        FROM engagement_sessions 
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    ''', conn)
    
    # AI predictions summary
    try:
        ai_stats = pd.read_sql_query('''
            SELECT 
                ai_prediction,
                COUNT(*) as count,
                AVG(ai_confidence) as avg_confidence
            FROM engagement_sessions 
            WHERE ai_prediction IS NOT NULL AND ai_prediction != ''
            GROUP BY ai_prediction
        ''', conn)
    except:
        ai_stats = pd.DataFrame()
    
    # Student-wise statistics  
    student_stats = pd.read_sql_query('''
        SELECT 
            s.name,
            s.student_id,
            AVG(e.engagement_score) as avg_engagement,
            COUNT(e.id) as total_sessions,
            MAX(e.timestamp) as last_seen
        FROM students s
        LEFT JOIN engagement_sessions e ON s.student_id = e.student_id
        GROUP BY s.student_id, s.name
        ORDER BY avg_engagement DESC
    ''', conn)
    
    conn.close()
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        if not engagement_df.empty:
            st.subheader("📅 Daily Engagement Trends")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(engagement_df['date'], engagement_df['avg_engagement'], marker='o', color='blue')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Engagement Score')
            ax.set_title('Daily Average Engagement')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No engagement data available yet.")
    
    with col2:
        if not ai_stats.empty:
            st.subheader("🤖 AI Predictions Summary")
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = {'engaged': 'green', 'not_engaged': 'red', 'neutral': 'orange'}
            bar_colors = [colors.get(pred, 'gray') for pred in ai_stats['ai_prediction']]
            ax.bar(ai_stats['ai_prediction'], ai_stats['count'], color=bar_colors)
            ax.set_xlabel('AI Prediction')
            ax.set_ylabel('Count')
            ax.set_title('AI Prediction Distribution')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No AI prediction data available yet.")
    
    # Student ranking
    if not student_stats.empty:
        st.subheader("🏆 Student Engagement Ranking")
        valid_stats = student_stats.dropna(subset=['avg_engagement'])
        if not valid_stats.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(valid_stats['name'], valid_stats['avg_engagement'])
            ax.set_xlabel('Average Engagement Score')
            ax.set_title('Student Engagement Comparison')
            st.pyplot(fig)
        else:
            st.info("No engagement data available for students yet.")
    
    # Data export
    st.subheader("📁 Export Data for Analysis")
    
    if st.button("Generate Enhanced Dataset"):
        conn = sqlite3.connect('student_engagement.db')
        
        full_data = pd.read_sql_query('''
            SELECT 
                e.timestamp,
                e.student_id,
                s.name as student_name,
                e.engagement_score,
                e.emotion,
                e.confidence,
                e.face_detected,
                e.ai_prediction,
                e.ai_confidence,
                CASE 
                    WHEN e.engagement_score >= 0.6 THEN 'Engaged'
                    WHEN e.engagement_score >= 0.3 THEN 'Neutral'
                    ELSE 'Disengaged'
                END as basic_category
            FROM engagement_sessions e
            LEFT JOIN students s ON e.student_id = s.student_id
            ORDER BY e.timestamp DESC
        ''', conn)
        
        conn.close()
        
        if not full_data.empty:
            csv = full_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Enhanced AI Dataset (CSV)",
                data=csv,
                file_name=f"ai_engagement_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success(f"✅ Enhanced dataset ready! {len(full_data)} records with AI predictions available.")
            st.dataframe(full_data.head(10), use_column_width=True)
        else:
            st.warning("No data available for export. Start a live session to collect data.")

# Footer
st.markdown("---")
st.markdown("""
### 🚀 AI-Enhanced Features:
- **Trained AI Model**: Custom engagement detection using your trained model
- **Continuous Real-time Processing**: Live video stream stays active until stopped  
- **Enhanced Analytics**: AI prediction tracking and confidence scores
- **Smart Fallbacks**: Graceful handling when AI model unavailable
- **Professional Interface**: Color-coded predictions and comprehensive stats

### 📊 Technical Details:
- **Face Recognition**: VGG-Face model via DeepFace
- **Emotion Detection**: Pre-trained CNN models  
- **AI Engagement**: Your custom trained Random Forest/Gradient Boosting model
- **Real-time Processing**: Continuous 30 FPS video stream with live annotations
""")

# Cleanup on app shutdown
import atexit
def cleanup():
    if 'video_processor' in st.session_state:
        st.session_state.video_processor.stop_camera()

atexit.register(cleanup)