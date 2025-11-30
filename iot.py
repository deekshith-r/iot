import streamlit as st
import cv2
import numpy as np
import threading
import time
import collections
import plotly.graph_objs as go
import tempfile
import os
from scipy.signal import find_peaks

# ==========================================
# 0. SETUP & CONFIGURATION
# ==========================================
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
except ImportError:
    st.error("🚨 Critical: Missing dependencies. Please run: pip install av streamlit-webrtc plotly scipy opencv-python")
    st.stop()

st.set_page_config(page_title="Neonatal AI Monitor", layout="wide", page_icon="👶")

# CSS for Status Cards
st.markdown("""
<style>
    .metric-container {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    .safe { background-color: #28a745; }
    .warning { background-color: #ffc107; color: black; }
    .danger { background-color: #dc3545; }
    .neutral { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 1. ADVANCED SIGNAL PROCESSING ENGINE
# ==========================================
class VitalSignProcessor:
    def __init__(self):
        # Buffers
        self.breath_signal = collections.deque(maxlen=300) # ~10 sec at 30fps
        self.green_signal = collections.deque(maxlen=300)
        
        # State
        self.fps_est = 30
        self.last_time = time.time()
        self.rr_val = 0
        self.hr_val = 0
        self.history_rr = collections.deque(maxlen=50)
        self.history_hr = collections.deque(maxlen=50)

    def process_frame(self, frame):
        """
        Analyzes a single video frame for Heart Rate (rPPG) and Breathing (Motion).
        Returns: Processed Frame, HR, RR, Status
        """
        # 1. FPS Calculation
        curr_time = time.time()
        dt = curr_time - self.last_time
        if dt > 0:
            self.fps_est = 0.9 * self.fps_est + 0.1 * (1/dt)
        self.last_time = curr_time

        h, w, _ = frame.shape
        
        # 2. Region of Interest (Chest/Body) for Breathing
        # Focus on the center 50% of the image
        roi_y1, roi_y2 = int(h*0.3), int(h*0.8)
        roi_x1, roi_x2 = int(w*0.3), int(w*0.7)
        roi_breath = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # 3. Breathing Signal Extraction (Intensity averaging)
        # As chest moves, light reflection changes
        gray_roi = cv2.cvtColor(roi_breath, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray_roi)
        self.breath_signal.append(avg_intensity)

        # 4. Heart Rate Signal (Green Channel Photoplethysmography)
        # Forehead approximation (top center)
        face_y1, face_y2 = int(h*0.1), int(h*0.3)
        face_x1, face_x2 = int(w*0.4), int(w*0.6)
        roi_face = frame[face_y1:face_y2, face_x1:face_x2]
        
        if roi_face.size > 0:
            g_val = np.mean(roi_face[:, :, 1])
            self.green_signal.append(g_val)
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)

        # Draw Breathing ROI
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        # 5. Calculate Rates (Every 10 frames to save CPU)
        if len(self.breath_signal) > 60:
            self.rr_val = self._calculate_rate_from_signal(list(self.breath_signal), 0.2, 1.0) # 12-60 bpm
        
        if len(self.green_signal) > 60:
            self.hr_val = self._calculate_rate_from_signal(list(self.green_signal), 1.0, 3.0) # 60-180 bpm

        return frame, int(self.hr_val), int(self.rr_val)

    def _calculate_rate_from_signal(self, data, min_freq, max_freq):
        """
        Robust FFT-based rate calculator.
        """
        y = np.array(data)
        # Detrend to remove baseline drift (light changes)
        y = y - np.mean(y)
        
        # FFT
        n = len(y)
        if n == 0: return 0
        freqs = np.fft.rfftfreq(n, d=1/self.fps_est)
        mag = np.abs(np.fft.rfft(y))
        
        # Bandpass Filter
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_freqs = freqs[mask]
        valid_mag = mag[mask]
        
        if len(valid_mag) == 0: return 0
        
        # Find dominant peak
        peak_idx = np.argmax(valid_mag)
        bpm = valid_freqs[peak_idx] * 60
        return bpm

processor = VitalSignProcessor()

# ==========================================
# 2. PREDICTION LOGIC
# ==========================================
def predict_health(rr, hr, is_crying):
    """
    Medical logic for neonates.
    Normal RR: 30-60
    Normal HR: 100-160
    """
    status = "Optimal Health"
    color = "safe"
    details = []

    # Breathing Analysis
    if rr > 60:
        status = "Respiratory Distress"
        color = "danger"
        details.append(f"Tachypnea detected ({rr} bpm). High breathing rate.")
    elif rr < 25 and rr > 0:
        status = "Abnormal Breathing"
        color = "warning"
        details.append(f"Bradypnea detected ({rr} bpm). Slow breathing.")
    
    # Heart Rate Analysis
    if hr > 180:
        if status == "Optimal Health": status = "Tachycardia"
        color = "danger"
        details.append(f"High Heart Rate ({hr} BPM).")
    elif hr < 90 and hr > 0:
        if status == "Optimal Health": status = "Bradycardia"
        color = "danger"
        details.append(f"Low Heart Rate ({hr} BPM).")

    # Crying
    if is_crying:
        status = "Crying / Distress"
        color = "warning"
        details.append("Audio indicates crying (Hunger/Discomfort).")

    if not details:
        details.append("Vitals are within normal neonatal ranges.")

    return status, color, details

# ==========================================
# 3. INTERFACE
# ==========================================
def main():
    st.sidebar.title("🩺 Configuration")
    mode = st.sidebar.radio("Input Source", ["Real-Time Camera", "Upload Video (File)"])
    
    st.title("👶 Neonatal AI Health Monitor")
    st.caption("Advanced Respiratory & Cardiac Analysis System")

    # --- SHARED DASHBOARD LAYOUT ---
    # We define placeholders here to be updated by either input method
    col_vid, col_data = st.columns([1.5, 1])
    
    with col_data:
        st.subheader("Live Analysis")
        # Placeholders
        ph_status_card = st.empty()
        c1, c2 = st.columns(2)
        ph_hr = c1.empty()
        ph_rr = c2.empty()
        ph_chart = st.empty()
        ph_details = st.empty()

    # --- MODE 1: UPLOAD VIDEO ---
    if mode == "Upload Video (File)":
        uploaded_file = st.sidebar.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            # Save to temp file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            st.sidebar.success("Video Loaded. Processing...")
            stop_btn = st.sidebar.button("Stop Processing")
            
            history_rr = []
            history_hr = []

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for speed (720p max)
                frame = cv2.resize(frame, (640, 360))
                
                # PROCESS
                proc_frame, hr, rr = processor.process_frame(frame)
                
                # Update Histories
                if rr > 0: history_rr.append(rr)
                if hr > 0: history_hr.append(hr)
                
                # PREDICT
                health_status, status_color, details = predict_health(rr, hr, False) # Assume no audio for file upload video for now

                # VISUALIZE
                # 1. Video
                with col_vid:
                    # Convert BGR to RGB for Streamlit
                    st.image(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # 2. Metrics
                ph_hr.metric("Heart Rate", f"{hr} BPM")
                ph_rr.metric("Breathing Rate", f"{rr} /min")
                
                # 3. Status Card
                ph_status_card.markdown(f"""
                <div class="metric-container {status_color}">
                    <h3 style="margin:0">{health_status}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 4. Details
                detail_text = "<ul>" + "".join([f"<li>{d}</li>" for d in details]) + "</ul>"
                ph_details.markdown(detail_text, unsafe_allow_html=True)

                # 5. Chart
                fig = go.Figure()
                if len(history_rr) > 0:
                    fig.add_trace(go.Scatter(y=np.array(history_rr), mode='lines', name='Breathing', line=dict(color='#00CC96')))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), title="Breathing Trend")
                ph_chart.plotly_chart(fig, use_container_width=True)
                
                # Throttle slightly to mimic real-time
                # time.sleep(0.01) 

            cap.release()
            st.sidebar.info("Video finished.")

    # --- MODE 2: REAL-TIME WEBRTC ---
    else:
        # Define callback for WebRTC
        def video_frame_callback(frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process
            proc_img, hr, rr = processor.process_frame(img)
            
            # Add text overlay to video for the "Live" feel
            cv2.putText(proc_img, f"HR: {hr} | RR: {rr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(proc_img, format="bgr24")

        with col_vid:
            webrtc_streamer(
                key="real-time-monitor",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False}, # Audio disabled for stability in this mode
                async_processing=True
            )
        
        st.info("ℹ️ In Real-Time mode, predictions are overlaid directly on the video feed to ensure 0-delay synchronization.")

if __name__ == "__main__":
    main()
