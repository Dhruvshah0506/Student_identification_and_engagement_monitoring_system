# Fixed Advanced Classroom Monitoring Dashboard
# Enhanced UI with compatibility for existing database schema
# Based on the original enhanced_app.py backend logic

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os
from datetime import datetime, timedelta
import json

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Configure page with dark theme
st.set_page_config(
    page_title="Classroom Monitoring AI Analytics", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .student-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background: #2d3748;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px;
        border-left: 4px solid #4299e1;
    }
    
    .engagement-high { border-left-color: #48bb78; }
    .engagement-medium { border-left-color: #ed8936; }
    .engagement-low { border-left-color: #f56565; }
    
    .stMetric > div > div > div > div {
        background-color: #2d3748;
        color: white;
        padding: 10px;
        border-radius: 8px;
    }
    
    .sidebar .sidebar-content {
        background-color: #1a202c;
    }
    
    .stSelectbox > div > div {
        background-color: #2d3748;
        color: white;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #48bb78; }
    .status-offline { background-color: #f56565; }
    
    .chart-container {
        background-color: #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Database setup with migration support
def init_database():
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    # Students table (original schema)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Engagement sessions table (original schema)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS engagement_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engagement_score REAL,
            emotion TEXT,
            confidence REAL,
            face_detected BOOLEAN,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # Check if new columns exist, if not add them
    try:
        cursor.execute("SELECT attention_level FROM engagement_sessions LIMIT 1")
    except sqlite3.OperationalError:
        # Add new columns to existing table
        cursor.execute("ALTER TABLE engagement_sessions ADD COLUMN attention_level TEXT")
        cursor.execute("ALTER TABLE engagement_sessions ADD COLUMN participation_score REAL")
        print("Database schema updated with new columns")
    
    conn.commit()
    conn.close()

init_database()

# Initialize session state with enhanced tracking
if "dashboard_state" not in st.session_state:
    st.session_state.dashboard_state = {
        "active_students": {},
        "current_detections": [],
        "session_analytics": {
            "total_frames": 0,
            "engaged_students": 0,
            "total_students": 0,
            "average_engagement": 0.0,
            "session_start": datetime.now()
        },
        "live_mode": True
    }

# Load models (same as original)
@st.cache_resource(show_spinner=False)
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade

face_cascade = load_models()

# Face recognition functions (same as original but with error handling)
def register_student(name, student_id, image):
    """Register a new student with their face encoding"""
    if not DEEPFACE_AVAILABLE:
        return False, "DeepFace not available"
    
    try:
        temp_path = f"temp_{student_id}.jpg"
        cv2.imwrite(temp_path, image)
        
        embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face")
        
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO students (name, student_id, face_encoding)
            VALUES (?, ?, ?)
        ''', (name, student_id, str(embedding[0]['embedding'])))
        
        conn.commit()
        conn.close()
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return True, "Student registered successfully"
    except Exception as e:
        if os.path.exists(f"temp_{student_id}.jpg"):
            os.remove(f"temp_{student_id}.jpg")
        return False, f"Registration failed: {str(e)}"

def identify_student(image):
    """Identify student from face"""
    if not DEEPFACE_AVAILABLE:
        return None, 0.0
    
    try:
        temp_path = "temp_identify.jpg"
        cv2.imwrite(temp_path, image)
        
        conn = sqlite3.connect('student_engagement.db')
        cursor = conn.cursor()
        cursor.execute('SELECT student_id, name, face_encoding FROM students')
        students = cursor.fetchall()
        conn.close()
        
        if not students:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None, 0.0
        
        current_embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face")
        
        best_match = None
        best_distance = float('inf')
        
        for student_id, name, encoding_str in students:
            try:
                stored_embedding = eval(encoding_str)
                distance = np.linalg.norm(np.array(current_embedding[0]['embedding']) - np.array(stored_embedding))
                
                if distance < best_distance and distance < 0.6:
                    best_distance = distance
                    best_match = (student_id, name)
            except:
                continue
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if best_match:
            confidence = max(0, 1 - best_distance)
            return best_match, confidence
        
        return None, 0.0
        
    except Exception as e:
        if os.path.exists("temp_identify.jpg"):
            os.remove("temp_identify.jpg")
        return None, 0.0

def detect_emotion_engagement(image):
    """Detect emotion and compute comprehensive engagement metrics"""
    if not DEEPFACE_AVAILABLE:
        return "neutral", 0.5, "Medium", 0.5
    
    try:
        temp_path = "temp_emotion.jpg"
        cv2.imwrite(temp_path, image)
        
        result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        
        # Enhanced engagement scoring
        engagement_score = 0.0
        
        # Positive engagement emotions
        engagement_score += emotions.get('happy', 0) * 0.8
        engagement_score += emotions.get('surprise', 0) * 0.6
        
        # Neutral engagement
        engagement_score += emotions.get('neutral', 0) * 0.5
        
        # Negative engagement emotions
        engagement_score += emotions.get('sad', 0) * 0.2
        engagement_score += emotions.get('angry', 0) * 0.1
        engagement_score += emotions.get('fear', 0) * 0.1
        engagement_score += emotions.get('disgust', 0) * 0.1
        
        engagement_score = engagement_score / 100.0
        
        # Attention level calculation
        if engagement_score >= 0.7:
            attention_level = "High"
        elif engagement_score >= 0.4:
            attention_level = "Medium"
        else:
            attention_level = "Low"
        
        # Participation score (based on engagement and emotion intensity)
        participation_score = engagement_score * 0.7 + (sum(emotions.values()) / 700.0) * 0.3
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return dominant_emotion, engagement_score, attention_level, participation_score
        
    except Exception as e:
        if os.path.exists("temp_emotion.jpg"):
            os.remove("temp_emotion.jpg")
        return "neutral", 0.5, "Medium", 0.5

def log_engagement_data(student_id, engagement_score, emotion, confidence, face_detected, attention_level=None, participation_score=None):
    """Enhanced logging with backward compatibility"""
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    # Check if new columns exist
    try:
        cursor.execute("SELECT attention_level, participation_score FROM engagement_sessions LIMIT 1")
        has_new_columns = True
    except sqlite3.OperationalError:
        has_new_columns = False
    
    if has_new_columns:
        cursor.execute('''
            INSERT INTO engagement_sessions 
            (student_id, engagement_score, emotion, confidence, face_detected, attention_level, participation_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (student_id, engagement_score, emotion, confidence, face_detected, attention_level, participation_score))
    else:
        # Fall back to original schema
        cursor.execute('''
            INSERT INTO engagement_sessions 
            (student_id, engagement_score, emotion, confidence, face_detected)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_id, engagement_score, emotion, confidence, face_detected))
    
    conn.commit()
    conn.close()

# Advanced Analytics Functions with backward compatibility
def get_real_time_analytics():
    """Get comprehensive real-time analytics with backward compatibility"""
    conn = sqlite3.connect('student_engagement.db')
    
    # Check if new columns exist
    try:
        conn.execute("SELECT attention_level FROM engagement_sessions LIMIT 1")
        has_new_columns = True
    except sqlite3.OperationalError:
        has_new_columns = False
    
    # Recent engagement data (last 30 minutes)
    if has_new_columns:
        recent_query = '''
            SELECT *, attention_level, participation_score FROM engagement_sessions 
            WHERE timestamp >= datetime('now', '-30 minutes')
            AND face_detected = 1
            ORDER BY timestamp DESC
        '''
    else:
        recent_query = '''
            SELECT * FROM engagement_sessions 
            WHERE timestamp >= datetime('now', '-30 minutes')
            AND face_detected = 1
            ORDER BY timestamp DESC
        '''
    
    recent_data = pd.read_sql_query(recent_query, conn)
    
    # Student-wise current status
    if has_new_columns:
        status_query = '''
            SELECT 
                s.name,
                s.student_id,
                e.engagement_score,
                e.emotion,
                e.attention_level,
                e.participation_score,
                e.timestamp
            FROM students s
            LEFT JOIN (
                SELECT student_id, engagement_score, emotion, attention_level, 
                       participation_score, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY timestamp DESC) as rn
                FROM engagement_sessions 
                WHERE timestamp >= datetime('now', '-5 minutes')
                AND face_detected = 1
            ) e ON s.student_id = e.student_id AND e.rn = 1
        '''
    else:
        status_query = '''
            SELECT 
                s.name,
                s.student_id,
                e.engagement_score,
                e.emotion,
                e.timestamp
            FROM students s
            LEFT JOIN (
                SELECT student_id, engagement_score, emotion, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY timestamp DESC) as rn
                FROM engagement_sessions 
                WHERE timestamp >= datetime('now', '-5 minutes')
                AND face_detected = 1
            ) e ON s.student_id = e.student_id AND e.rn = 1
        '''
    
    current_status = pd.read_sql_query(status_query, conn)
    
    # Add computed columns if they don't exist
    if not has_new_columns and not current_status.empty:
        # Compute attention level from engagement score
        current_status['attention_level'] = current_status['engagement_score'].apply(
            lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low' if pd.notnull(x) else 'Unknown'
        )
        current_status['participation_score'] = current_status['engagement_score'] * 0.8
    
    # Hourly engagement trends
    hourly_trends = pd.read_sql_query('''
        SELECT 
            strftime('%H:%M', timestamp) as time_slot,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as detections
        FROM engagement_sessions 
        WHERE date(timestamp) = date('now')
        AND face_detected = 1
        GROUP BY strftime('%H:%M', timestamp)
        ORDER BY time_slot
    ''', conn)
    
    # Emotion distribution
    emotion_dist = pd.read_sql_query('''
        SELECT 
            emotion,
            COUNT(*) as count,
            AVG(engagement_score) as avg_engagement
        FROM engagement_sessions 
        WHERE timestamp >= datetime('now', '-30 minutes')
        AND face_detected = 1
        GROUP BY emotion
    ''', conn)
    
    conn.close()
    
    return recent_data, current_status, hourly_trends, emotion_dist

# UI Components
def render_header():
    """Render the main dashboard header"""
    st.markdown("""
        <div class="main-header">
            <h1>🎓 CLASSROOM MONITORING</h1>
            <h2>AI ANALYTICS</h2>
        </div>
    """, unsafe_allow_html=True)

def render_student_detection_cards(detections_data):
    """Render individual student detection cards"""
    if detections_data.empty:
        st.info("No active student detections")
        return
    
    # Create columns for student cards
    cols = st.columns(min(len(detections_data), 3))
    
    for idx, (_, student) in enumerate(detections_data.iterrows()):
        col_idx = idx % 3
        
        with cols[col_idx]:
            engagement_score = student.get('engagement_score', 0)
            participation_score = student.get('participation_score', engagement_score * 0.8)
            
            st.markdown(f"""
                <div class="student-card">
                    <h4>{student.get('name', 'Unknown Student')}</h4>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <div>
                            <strong>Attention level:</strong><br>
                            <span style="font-size: 1.2em; color: #4fd1c7;">
                                {int(engagement_score * 100)}%
                            </span>
                        </div>
                        <div>
                            <strong>Participation:</strong><br>
                            <span style="font-size: 1.2em; color: #4fd1c7;">
                                {int(participation_score * 100)}
                            </span>
                        </div>
                    </div>
                    <div style="margin: 10px 0;">
                        <strong>Emotion:</strong> {student.get('emotion', 'Unknown').title()}
                    </div>
                    <div style="margin: 10px 0;">
                        <strong>Score:</strong> {int(engagement_score * 100)}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_engagement_charts(hourly_trends, emotion_dist, current_status):
    """Render interactive engagement charts"""
    
    # Engagement Over Time Chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Engagement Over Time")
        
        if not hourly_trends.empty:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=hourly_trends['time_slot'],
                y=hourly_trends['avg_engagement'],
                mode='lines+markers',
                name='Average Engagement',
                line=dict(color='#4fd1c7', width=3),
                marker=dict(size=8, color='#4fd1c7')
            ))
            fig_line.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(gridcolor='#4a5568'),
                yaxis=dict(gridcolor='#4a5568', range=[0, 1])
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No engagement data available yet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Class Engagement")
        
        # Calculate engagement distribution
        if not current_status.empty and 'engagement_score' in current_status.columns:
            valid_scores = current_status.dropna(subset=['engagement_score'])
            if not valid_scores.empty:
                engaged_count = len(valid_scores[valid_scores['engagement_score'] > 0.6])
                neutral_count = len(valid_scores[(valid_scores['engagement_score'] >= 0.3) & (valid_scores['engagement_score'] <= 0.6)])
                low_count = len(valid_scores[valid_scores['engagement_score'] < 0.3])
                
                if engaged_count + neutral_count + low_count > 0:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['High', 'Medium', 'Low'],
                        values=[engaged_count, neutral_count, low_count],
                        marker_colors=['#48bb78', '#ed8936', '#f56565']
                    )])
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        showlegend=True
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No valid engagement data available")
            else:
                st.info("No valid engagement scores available")
        else:
            st.info("No engagement data available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom row charts
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Attention Level")
        
        if not current_status.empty and 'attention_level' in current_status.columns:
            attention_data = current_status.dropna(subset=['attention_level'])
            if not attention_data.empty:
                attention_counts = attention_data['attention_level'].value_counts()
                fig_bar1 = go.Figure([go.Bar(
                    x=attention_counts.index,
                    y=attention_counts.values,
                    marker_color=['#48bb78', '#ed8936', '#f56565']
                )])
                fig_bar1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis=dict(gridcolor='#4a5568'),
                    yaxis=dict(gridcolor='#4a5568')
                )
                st.plotly_chart(fig_bar1, use_container_width=True)
            else:
                st.info("No attention level data available")
        else:
            st.info("No attention level data available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎭 Participation Score")
        
        if not current_status.empty and 'participation_score' in current_status.columns:
            participation_data = current_status.dropna(subset=['participation_score'])
            if not participation_data.empty:
                fig_bar2 = go.Figure([go.Bar(
                    x=participation_data['name'],
                    y=participation_data['participation_score'],
                    marker_color='#4fd1c7'
                )])
                fig_bar2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis=dict(gridcolor='#4a5568', tickangle=45),
                    yaxis=dict(gridcolor='#4a5568')
                )
                st.plotly_chart(fig_bar2, use_container_width=True)
            else:
                st.info("No participation data available")
        else:
            st.info("No participation data available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("😊 Emotion Analysis")
        
        if not emotion_dist.empty:
            fig_pie2 = go.Figure(data=[go.Pie(
                labels=emotion_dist['emotion'],
                values=emotion_dist['count'],
                textinfo='label+percent',
                textposition='inside'
            )])
            fig_pie2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig_pie2, use_container_width=True)
        else:
            st.info("No emotion data available")
        st.markdown('</div>', unsafe_allow_html=True)

# Main Application
def main():
    render_header()
    
    # Sidebar controls (minimal, as requested collapsed initially)
    with st.sidebar:
        st.header("🔧 Controls")
        mode = st.selectbox("Mode", ["Live Dashboard", "Student Registration", "Settings"])
        
        if mode == "Live Dashboard":
            st.subheader("Detection Settings")
            confidence_threshold = st.slider("Recognition Confidence", 0.1, 1.0, 0.6)
            engagement_threshold = st.slider("Engagement Threshold", 0.1, 1.0, 0.5)
            
            # Manual refresh button instead of auto-refresh to avoid issues
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
        elif mode == "Student Registration":
            st.subheader("Register New Student")
            new_name = st.text_input("Student Name")
            new_id = st.text_input("Student ID")
    
    if mode == "Live Dashboard":
        # Camera input section
        st.subheader("📷 Live Camera Feed")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img = st.camera_input("Capture frame for analysis", key="main_camera")
            
            if img is not None:
                # Process image (same logic as original)
                pil_image = Image.open(img)
                rgb_image = np.array(pil_image)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                
                current_detections = []
                
                for face in faces:
                    x, y, w, h = face
                    face_region = bgr_image[y:y+h, x:x+w]
                    
                    # Identify student
                    student_info, recognition_confidence = identify_student(face_region)
                    
                    # Detect emotion and engagement
                    emotion, engagement_score, attention_level, participation_score = detect_emotion_engagement(face_region)
                    
                    if student_info and recognition_confidence >= confidence_threshold:
                        student_id, student_name = student_info
                        
                        current_detections.append({
                            'name': student_name,
                            'student_id': student_id,
                            'engagement_score': engagement_score,
                            'emotion': emotion,
                            'attention_level': attention_level,
                            'participation_score': participation_score,
                            'confidence': recognition_confidence
                        })
                        
                        # Log data
                        log_engagement_data(student_id, engagement_score, emotion, 
                                          recognition_confidence, True, attention_level, participation_score)
                    else:
                        # Log unknown student
                        log_engagement_data("unknown", engagement_score, emotion, 
                                          0.0, True, attention_level, participation_score)
                
                # Update session state
                st.session_state.dashboard_state["current_detections"] = current_detections
                
                # Draw annotations
                annotated_image = bgr_image.copy()
                for i, face in enumerate(faces):
                    x, y, w, h = face
                    detection = current_detections[i] if i < len(current_detections) else {}
                    
                    engagement = detection.get('engagement_score', 0)
                    color = (0, 255, 0) if engagement >= engagement_threshold else (0, 0, 255)
                    
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                    
                    label = detection.get('name', 'Unknown')
                    cv2.putText(annotated_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    emotion_label = f"{detection.get('emotion', 'unknown').title()}"
                    cv2.putText(annotated_image, emotion_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Live Detection Results")
        
        with col2:
            # Get real-time analytics
            recent_data, current_status, hourly_trends, emotion_dist = get_real_time_analytics()
            
            # Render student detection cards
            st.subheader("👥 Active Student Detections")
            render_student_detection_cards(pd.DataFrame(st.session_state.dashboard_state["current_detections"]))
        
        # Render charts
        st.markdown("---")
        render_engagement_charts(hourly_trends, emotion_dist, current_status)
    
    elif mode == "Student Registration":
        st.subheader("👤 Register New Student")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            registration_img = st.camera_input("Capture student photo", key="registration_camera")
            
            if registration_img is not None and new_name and new_id:
                if st.button("Register Student"):
                    pil_image = Image.open(registration_img)
                    rgb_image = np.array(pil_image)
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    
                    success, message = register_student(new_name, new_id, bgr_image)
                    
                    if success:
                        st.success(message)
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

if __name__ == "__main__":
    main()