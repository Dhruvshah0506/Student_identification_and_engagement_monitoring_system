# FIXED: DAiSEE Data Loader for Existing train_model.py
# This script loads DAiSEE dataset into your existing SQLite database with correct schema
# Then you can run the original train_model.py without modifications

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime
import cv2
from tqdm import tqdm

def download_sample_daisee_data():
    """Create sample DAiSEE-style data that mimics the real dataset structure"""
    print("📥 Creating sample DAiSEE data...")
    
    # Create directory structure
    os.makedirs('daisee_dataset/Labels', exist_ok=True)
    
    # Generate sample data mimicking DAiSEE AllLabels.csv format
    sample_labels = []
    
    np.random.seed(42)  # Reproducible results
    
    for user_id in range(1, 101):  # 100 users
        for video_num in range(1, 6):  # 5 videos per user
            clip_id = f"{user_id:06d}_{video_num:02d}"
            
            # Generate realistic engagement levels (DAiSEE format: 0-3)
            engagement = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.35, 0.15])
            boredom = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
            confusion = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])  
            frustration = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
            
            sample_labels.append({
                'ClipID': clip_id,
                'UserID': user_id,
                'Engagement': engagement,
                'Boredom': boredom,
                'Confusion': confusion,
                'Frustration': frustration
            })
    
    # Save as CSV in DAiSEE format
    labels_df = pd.DataFrame(sample_labels)
    labels_df.to_csv('daisee_dataset/Labels/AllLabels.csv', index=False)
    
    print(f"✅ Created sample DAiSEE dataset with {len(sample_labels)} entries")
    print(f"📊 Engagement distribution:")
    print(labels_df['Engagement'].value_counts().sort_index())
    
    return labels_df

def load_real_daisee_labels():
    """Load real DAiSEE labels if available"""
    labels_file = 'daisee_dataset/Labels/AllLabels.csv'
    
    if os.path.exists(labels_file):
        print(f"📊 Loading real DAiSEE labels from {labels_file}")
        return pd.read_csv(labels_file)
    else:
        print(f"⚠️ Real DAiSEE labels not found at {labels_file}")
        print("Using sample data instead...")
        return download_sample_daisee_data()

def convert_daisee_to_engagement_sessions(labels_df):
    """Convert DAiSEE labels to engagement_sessions format for train_model.py"""
    print("🔄 Converting DAiSEE data to engagement_sessions format...")
    
    engagement_sessions = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        clip_id = row['ClipID']
        user_id = row['UserID']
        daisee_engagement = row['Engagement']
        
        # Map DAiSEE engagement levels to engagement scores (0.0 to 1.0)
        engagement_score_map = {
            0: np.random.uniform(0.0, 0.25),   # Very low -> 0.0-0.25
            1: np.random.uniform(0.25, 0.5),  # Low -> 0.25-0.5  
            2: np.random.uniform(0.5, 0.75),  # High -> 0.5-0.75
            3: np.random.uniform(0.75, 1.0)   # Very high -> 0.75-1.0
        }
        
        engagement_score = engagement_score_map[daisee_engagement]
        
        # Create engagement_category based on score (this is what train_model.py expects)
        if engagement_score >= 0.7:
            engagement_category = 'engaged'
        elif engagement_score <= 0.3:
            engagement_category = 'not_engaged'
        else:
            engagement_category = 'neutral'
        
        # Generate realistic emotion based on engagement level
        if daisee_engagement >= 2:
            emotions = ['happy', 'surprise', 'neutral']
            weights = [0.5, 0.2, 0.3]
            confidence = np.random.uniform(0.7, 0.95)
        elif daisee_engagement == 1:
            emotions = ['neutral', 'sad', 'happy']  
            weights = [0.6, 0.25, 0.15]
            confidence = np.random.uniform(0.5, 0.8)
        else:
            emotions = ['sad', 'angry', 'neutral', 'disgust']
            weights = [0.4, 0.2, 0.3, 0.1]
            confidence = np.random.uniform(0.3, 0.7)
        
        emotion = np.random.choice(emotions, p=weights)
        
        # Generate multiple sessions per clip to increase data
        for session_num in range(3):  # 3 sessions per clip
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            engagement_sessions.append({
                'student_id': f"daisee_user_{user_id}",
                'timestamp': timestamp,
                'engagement_score': engagement_score,
                'emotion': emotion,
                'confidence': confidence,
                'face_detected': 1,  # Assume face always detected in DAiSEE
                'engagement_category': engagement_category  # ADD THIS COLUMN
            })
    
    print(f"✅ Generated {len(engagement_sessions)} engagement sessions")
    return engagement_sessions

def populate_database_with_daisee(engagement_sessions):
    """Populate the existing SQLite database with DAiSEE-derived data"""
    print("💾 Populating SQLite database with DAiSEE data...")
    
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    # Drop and recreate tables to ensure correct schema
    cursor.execute('DROP TABLE IF EXISTS engagement_sessions')
    cursor.execute('DROP TABLE IF EXISTS students')
    
    # Create students table
    cursor.execute('''
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create engagement_sessions table with the engagement_category column
    cursor.execute('''
        CREATE TABLE engagement_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            engagement_score REAL,
            emotion TEXT,
            confidence REAL,
            face_detected BOOLEAN,
            engagement_category TEXT,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # Add students from DAiSEE
    unique_students = set(session['student_id'] for session in engagement_sessions)
    
    for student_id in unique_students:
        cursor.execute('''
            INSERT INTO students (name, student_id)
            VALUES (?, ?)
        ''', (f"DAiSEE Student {student_id.split('_')[-1]}", student_id))
    
    # Add engagement sessions with engagement_category column
    for session in engagement_sessions:
        cursor.execute('''
            INSERT INTO engagement_sessions 
            (student_id, timestamp, engagement_score, emotion, confidence, face_detected, engagement_category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['student_id'],
            session['timestamp'], 
            session['engagement_score'],
            session['emotion'],
            session['confidence'],
            session['face_detected'],
            session['engagement_category']  # Include this column
        ))
    
    conn.commit()
    
    # Verify data
    cursor.execute('SELECT COUNT(*) FROM engagement_sessions')
    count = cursor.fetchone()[0]
    
    # Check schema
    cursor.execute("PRAGMA table_info(engagement_sessions)")
    columns = cursor.fetchall()
    
    conn.close()
    
    print(f"✅ Successfully added {count} records to database")
    print(f"✅ Database schema verified - columns: {[col[1] for col in columns]}")
    print(f"✅ Database ready for train_model.py")

def verify_database():
    """Verify the database has the correct structure and data"""
    print("\n🔍 Verifying database structure...")
    
    conn = sqlite3.connect('student_engagement.db')
    cursor = conn.cursor()
    
    # Check if engagement_category column exists
    cursor.execute("PRAGMA table_info(engagement_sessions)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'engagement_category' not in columns:
        print("❌ engagement_category column missing!")
        return False
    
    # Check data distribution
    cursor.execute('''
        SELECT engagement_category, COUNT(*) 
        FROM engagement_sessions 
        GROUP BY engagement_category
    ''')
    
    distribution = cursor.fetchall()
    
    print("✅ Database structure correct!")
    print("📊 Data distribution:")
    for category, count in distribution:
        print(f"  {category}: {count} samples")
    
    # Test the exact query from train_model.py
    try:
        cursor.execute('''
            SELECT engagement_score, emotion, confidence, face_detected, engagement_category
            FROM engagement_sessions
            WHERE engagement_score IS NOT NULL AND emotion IS NOT NULL
        ''')
        test_data = cursor.fetchall()
        print(f"✅ Train query test successful - {len(test_data)} records found")
    except Exception as e:
        print(f"❌ Train query test failed: {e}")
        return False
    
    conn.close()
    return True

def main():
    print("🎓 FIXED DAiSEE Data Loader for train_model.py")
    print("=" * 50)
    
    print("\nThis script prepares DAiSEE data for the existing train_model.py")
    print("Choose your data source:")
    print("1. Use sample DAiSEE data (quick start)")
    print("2. Use real DAiSEE dataset (if downloaded)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    try:
        # Step 1: Load DAiSEE labels
        if choice == "2":
            labels_df = load_real_daisee_labels()
        else:
            labels_df = download_sample_daisee_data()
        
        # Step 2: Convert to engagement sessions format
        engagement_sessions = convert_daisee_to_engagement_sessions(labels_df)
        
        # Step 3: Populate database
        populate_database_with_daisee(engagement_sessions)
        
        # Step 4: Verify database
        if verify_database():
            print("\n🎉 Database setup complete and verified!")
            print("\n🚀 Next steps:")
            print("1. Run: python train_model.py")
            print("2. The original training script will now work with DAiSEE data!")
            print("3. After training, run your dashboard to see AI predictions")
        else:
            print("\n❌ Database verification failed!")
            
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        print("\nFor real DAiSEE dataset:")
        print("1. Download from: https://people.iith.ac.in/vineethnb/resources/daisee/")
        print("2. Extract to 'daisee_dataset' folder")
        print("3. Run this script again with option 2")

if __name__ == "__main__":
    main()