import streamlit as st
import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import os
import tempfile
from datetime import datetime
import plotly.graph_objects as go

# Import your custom modules
from config import *
from eye_utils import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
from mouth_utils import preprocess_mouth, get_mouth_roi
from alert import AlertSystem

# --- PAGE CONFIG ---
st.set_page_config(page_title="Driver Drowsiness System", page_icon="🚗", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-title { font-size: 45px; font-weight: bold; text-align: left; margin-bottom: 20px; }
    .slider-label { font-size: 14px; color: #555; margin-bottom: -10px; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_resources():
    mouth_model = tf.keras.models.load_model(YAWN_MODEL_PATH)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    alert_sys = AlertSystem(ALARM_SOUND_PATH)
    return mouth_model, face_landmarker, alert_sys

mouth_model, face_landmarker, alert = load_resources()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Select Detection Mode", ["Real-Time Detection", "Video Upload Detection"])
    st.markdown("---")
    run_cam = st.checkbox("Start/Stop Camera", value=False)
    show_fps_sidebar = st.toggle("Show FPS", value=True)
    st.markdown("---")
    if st.button("Reset Session"):
        st.rerun()

# --- MAIN UI HEADER ---
st.markdown('<div class="main-title">🚗 Driver Drowsiness Detection System</div>', unsafe_allow_html=True)

# --- SHARED LOGIC ---
def process_frame(frame, ear_hist, mouth_hist, eye_start, ear_limit, yawn_limit):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    
    drowsy_detected = False
    eye_status, mouth_status = "AWAKE", "NORMAL"
    ear_val = 0.0
    
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        # EAR calculation
        left = [landmarks[i] for i in LEFT_EYE_IDX]
        right = [landmarks[i] for i in RIGHT_EYE_IDX]
        ear_val = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        ear_hist.append(ear_val)
        avg_ear = np.mean(ear_hist)
        
        # Yawn Detection
        m_roi, _ = get_mouth_roi(frame, landmarks, w, h)
        m_input = preprocess_mouth(m_roi)
        if m_input is not None:
            yawn_val = mouth_model.predict(m_input, verbose=0)[0][1]
            mouth_hist.append(yawn_val)
            avg_yawn = np.mean(mouth_hist)
            if avg_yawn > yawn_limit: mouth_status = "YAWNING"

        # Drowsiness logic
        if avg_ear < ear_limit:
            eye_status = "SLEEPY"
            if eye_start is None: eye_start = time.time()
            elif time.time() - eye_start > CLOSED_EYE_SECONDS: drowsy_detected = True
        else:
            eye_start = None

        # Overlay Text
        cv2.putText(frame, f"EYE: {eye_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"MOUTH: {mouth_status}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
    return frame, drowsy_detected, ear_val, eye_start

# --- MODE 1: REAL-TIME ---
if mode == "Real-Time Detection":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="slider-label">Eye Aspect Ratio (Sensitivity):</p>', unsafe_allow_html=True)
        ear_thresh = st.slider("", 0.10, 0.40, 0.23, 0.01, key="ear_slider")
    with col2:
        st.markdown('<p class="slider-label">Yawn Confidence (Threshold):</p>', unsafe_allow_html=True)
        yawn_thresh = st.slider("", 0.10, 0.90, 0.50, 0.05, key="yawn_slider")

    frame_placeholder = st.empty()
    
    if run_cam:
        cap = cv2.VideoCapture(0)
        
        # --- HARDWARE RESIZE ---
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        ear_history = deque(maxlen=EAR_HISTORY)
        yawn_history = deque(maxlen=MOUTH_HISTORY)
        eye_start_time = None
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- SOFTWARE SAFETY RESIZE ---
            frame = cv2.resize(frame, (1280, 720))
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            frame, drowsy, cur_ear, eye_start_time = process_frame(
                frame, ear_history, yawn_history, eye_start_time, ear_thresh, yawn_thresh
            )
            
            if drowsy: alert.play()

            if show_fps_sidebar:
                cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if not run_cam: break
        cap.release()

# --- MODE 2: VIDEO UPLOAD ---
elif mode == "Video Upload Detection":
    st.subheader("📁 Upload Video for Analysis")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        if st.button("Process Video"):
            cap = cv2.VideoCapture(tfile.name)
            ear_history = deque(maxlen=7)
            yawn_history = deque(maxlen=7)
            eye_start_time = None
            frame_win = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Resize video for consistent processing
                frame = cv2.resize(frame, (1280, 720))
                
                frame, drowsy, cur_ear, eye_start_time = process_frame(
                    frame, ear_history, yawn_history, eye_start_time, 0.23, 0.50
                )
                
                frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
            
# Add this at the very bottom of your script
st.divider()
with st.expander("🛡️ Ethical Disclosure & Privacy Policy"):
    st.write("""
    **Privacy-by-Design:** All video processing is performed locally. No data is stored or transmitted.
    
    **System Limitations:** Accuracy is dependent on lighting and camera quality. 
    Dark sunglasses or extreme head tilts may impact detection.
    
    **Legal Disclaimer:** This software is a prototype for educational/safety 
    assistance purposes. The developer is not liable for any incidents resulting 
    from the use or failure of this system.
    """)            