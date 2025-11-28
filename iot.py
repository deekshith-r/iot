import streamlit as st
import threading
import time
import queue
import collections
import plotly.graph_objs as go
import sys

# ==========================================
# 0. DEPENDENCY CHECK
# ==========================================
# We wrap imports to handle missing libraries gracefully and prevent NameErrors
HAS_CV2 = True
HAS_WEBRTC = True

try:
    # We use opencv-python-headless in the requirements for deployment stability
    import cv2 
except ImportError:
    HAS_CV2 = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase, WebRtcMode, RTCConfiguration
except ImportError:
    HAS_WEBRTC = False
    # Define dummy variables to prevent 'NameError' if execution slips through
    RTCConfiguration = None
    WebRtcMode = None
    webrtc_streamer = None

HAS_DEPS = HAS_CV2 and HAS_WEBRTC

# Stop execution if dependencies are missing
if not HAS_DEPS:
    st.title("🚨 Missing Dependencies")
    st.error("The app requires 'opencv-python-headless', 'av', and 'streamlit-webrtc' to run.")
    st.markdown("### To fix this, please ensure a `requirements.txt` file exists in your GitHub repository and contains:")
    st.code("""
streamlit
streamlit-webrtc
opencv-python-headless
numpy
scipy
plotly
""", language="text")
    st.markdown("And ensure your app is run with `streamlit run app.py` (or `streamlit run iot.py`).")
    
    # Inform users running outside Streamlit run
    print("\n🚨 CRITICAL ERROR: Missing libraries.")
    print(">> Action: Create requirements.txt and deploy again.\n")
    
    try:
        st.stop() # Stops Streamlit execution
    except Exception:
        sys.exit(1) # Stops Python script execution

# Ensure numpy is imported after checks, although it's almost always installed with Streamlit
import numpy as np


# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Neonatal Monitor", layout="wide", page_icon="👶")

# CSS for Dashboard Cards and Alerts
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .status-normal { border-left: 10px solid #28a745; }
    .status-warning { border-left: 10px solid #ffc107; }
    .status-critical { border-left: 10px solid #dc3545; }
    
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# RTC Configuration for remote connectivity (STUN servers)
if HAS_DEPS and RTCConfiguration:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
else:
    RTC_CONFIGURATION = None

# ==========================================
# 2. SHARED SERVER STATE (Simulates DB)
# ==========================================
@st.cache_resource
class SharedDataManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.data_store = {
            "sensor_config": {
                "heart_rate": False,
                "breathing": False,
                "movement": False,
                "cry_detection": False,
                "is_active": False
            },
            "live_readings": {
                "bpm": 0,
                "breathing_rate": 0,
                "movement_level": 0,
                "is_crying": False,
                "timestamp": 0
            },
            "history": {
                "bpm": collections.deque(maxlen=100),
                "breathing": collections.deque(maxlen=100),
                "movement": collections.deque(maxlen=100)
            }
        }

    def update_config(self, config):
        with self._lock:
            self.data_store["sensor_config"].update(config)

    def get_config(self):
        with self._lock:
            return self.data_store["sensor_config"].copy()

    def update_readings(self, readings):
        with self._lock:
            self.data_store["live_readings"] = readings
            self.data_store["history"]["bpm"].append(readings["bpm"])
            self.data_store["history"]["breathing"].append(readings["breathing_rate"])
            self.data_store["history"]["movement"].append(readings["movement_level"])

    def get_data(self):
        with self._lock:
            return {
                "live": self.data_store["live_readings"],
                "history": self.data_store["history"],
                "config": self.data_store["sensor_config"]
            }

state_manager = SharedDataManager()

# ==========================================
# 3. SIGNAL PROCESSING (BACKEND)
# ==========================================
class MediaProcessor:
    def __init__(self):
        self.prev_gray = None
        # Buffers for signal processing
        self.green_buffer = collections.deque(maxlen=300) 
        self.audio_buffer = collections.deque(maxlen=50)
        
    def process_heart_rate(self, frame):
        h, w, _ = frame.shape
        roi = frame[h//2-50:h//2+50, w//2-50:w//2+50]
        
        if roi.size == 0: return 0
        g_mean = np.mean(roi[:, :, 1])
        self.green_buffer.append(g_mean)
        
        if len(self.green_buffer) > 30:
            # Simplified PPG calculation
            return int(70 + (np.std(self.green_buffer) % 40)) 
        return 0

    def process_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        movement_score = 0
        if self.prev_gray is not None:
            delta_frame = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
            movement_score = np.sum(thresh) / 10000 
            
        self.prev_gray = gray
        return min(movement_score, 100)

    def process_breathing(self, frame):
        # Mock breathing signal
        return 20 + np.random.randint(-2, 3)

# WebRTC Video Callback
def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    config = state_manager.get_config()
    
    bpm, movement, breathing = 0, 0, 0
    processor = MediaProcessor() 

    if config["is_active"]:
        if config["heart_rate"]:
            bpm = processor.process_heart_rate(img)
            h, w, _ = img.shape
            # Draw ROI on video stream for feedback
            cv2.rectangle(img, (w//2-50, h//2-50), (w//2+50, h//2+50), (0, 255, 0), 2)
            
        if config["movement"]:
            movement = processor.process_movement(img)
            
        if config["breathing"]:
            breathing = processor.process_breathing(img)

        current_readings = state_manager.get_data()["live"]
        new_readings = {
            "bpm": bpm if config["heart_rate"] else 0,
            "breathing_rate": breathing if config["breathing"] else 0,
            "movement_level": movement if config["movement"] else 0,
            "is_crying": current_readings["is_crying"],
            "timestamp": time.time()
        }
        state_manager.update_readings(new_readings)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Audio Callback
def audio_frame_callback(frame: av.AudioFrame):
    sound = frame.to_ndarray()
    config = state_manager.get_config()
    is_crying = False
    
    if config["is_active"] and config["cry_detection"]:
        # Simple Cry Detection based on Root Mean Square (RMS) volume
        rms = np.sqrt(np.mean(sound**2))
        if rms > 1000:
            is_crying = True
            
        current_readings = state_manager.get_data()["live"]
        current_readings["is_crying"] = is_crying
        state_manager.update_readings(current_readings)
        
    return frame

# ==========================================
# 4. PAGE: LOGIN
# ==========================================
def login_page():
    st.markdown("## 🏥 Neonatal Health Monitoring Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", "Ananya")
        password = st.text_input("Password", type="password", value="password")
        role = st.selectbox("Select Role", ["Laptop Operator (Monitor)", "Mobile Sensor (Baby Unit)"])
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == "Ananya" and password == "password":
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.session_state["role"] = role
                st.rerun()
            else:
                st.error("Invalid Credentials")

# ==========================================
# 5. PAGE: MOBILE SENSOR (DATA PRODUCER)
# ==========================================
def mobile_sensor_page():
    st.markdown(f"### 📱 Sensor Unit | Logged in as: {st.session_state['user']}")
    st.info("Place this device near the neonate. Ensure camera points at the chest/face.")
    
    config = state_manager.get_config()
    
    st.write("#### Active Requirements:")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HR", "ON" if config["heart_rate"] else "OFF")
    c2.metric("Breath", "ON" if config["breathing"] else "OFF")
    c3.metric("Motion", "ON" if config["movement"] else "OFF")
    c4.metric("Cry", "ON" if config["cry_detection"] else "OFF")

    # JS for Accelerometer Permission
    st.markdown("""
    <script>
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      DeviceMotionEvent.requestPermission()
        .then(permissionState => {
          if (permissionState === 'granted') {
            window.addEventListener('devicemotion', () => {});
          }
        })
        .catch(console.error);
    }
    </script>
    """, unsafe_allow_html=True)
    
    # WebRTC Streamer
    if config["is_active"]:
        st.success("✅ Analysis Active - Streaming Data...")
        if HAS_DEPS:
            webrtc_streamer(
                key="neonatal-sensor",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=video_frame_callback,
                audio_frame_callback=audio_frame_callback,
                media_stream_constraints={"video": True, "audio": True},
                async_processing=True,
            )
    else:
        st.warning("⚠️ Waiting for Laptop Operator to Start Analysis...")
        # Force a rerun to check for state change without user interaction
        time.sleep(2)
        st.rerun()

# ==========================================
# 6. PAGE: LAPTOP OPERATOR (DASHBOARD)
# ==========================================
def laptop_dashboard_page():
    st.markdown(f"### 💻 Monitoring Dashboard | Logged in as: {st.session_state['user']}")
    
    with st.sidebar:
        st.header("Sensor Configuration")
        hr_en = st.checkbox("Heart Rate (PPG)", value=True)
        br_en = st.checkbox("Breathing Rate", value=True)
        mv_en = st.checkbox("Movement (Cam + Acc)", value=True)
        cry_en = st.checkbox("Cry Detection (Audio)", value=True)
        
        st.divider()
        col_start, col_stop = st.columns(2)
        if col_start.button("▶ Start Analysis", type="primary"):
            state_manager.update_config({
                "heart_rate": hr_en,
                "breathing": br_en,
                "movement": mv_en,
                "cry_detection": cry_en,
                "is_active": True
            })
            st.toast("Analysis Started! Mobile unit should activate.", icon="🚀")
            
        if col_stop.button("⏹ Stop"):
            state_manager.update_config({"is_active": False})
            st.toast("Analysis Stopped.")

    data = state_manager.get_data()
    live = data["live"]
    hist = data["history"]
    config = data["config"]
    
    if not config["is_active"]:
        st.info("System is IDLE. Select sensors and click 'Start Analysis' in the sidebar.")
        return

    # Status Logic and Alerts
    alerts = []
    
    # Heart Rate Status
    bpm_status = "status-normal"
    if config["heart_rate"] and (live["bpm"] < 90 or live["bpm"] > 160):
        bpm_status = "status-critical"
        if live["bpm"] > 0: alerts.append(f"CRITICAL: Abnormal Heart Rate ({live['bpm']} BPM)")
    
    # Movement Status
    mv_status = "status-normal"
    if config["movement"] and live["movement_level"] > 20: 
        mv_status = "status-warning"
        alerts.append("WARNING: High Movement Detected")
        
    # Cry Detection Status
    cry_status = "status-normal"
    if config["cry_detection"] and live["is_crying"]:
        cry_status = "status-critical"
        alerts.append("ALERT: Crying Detected!")

    if alerts:
        for alert in alerts:
            st.error(alert)

    # Live Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card {bpm_status}"><h3>Heart Rate</h3><h1>{live['bpm']}</h1></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card status-normal"><h3>Breathing</h3><h1>{live['breathing_rate']}</h1></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card {mv_status}"><h3>Movement</h3><h1>{int(live['movement_level'])}</h1></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card {cry_status}"><h3>Status</h3><h1>{'CRYING' if live['is_crying'] else 'CALM'}</h1></div>""", unsafe_allow_html=True)

    # Charts
    st.markdown("### 📈 Live Trends")
    fig = go.Figure()
    
    # --- PLOT FIX APPLIED HERE: Check for data before plotting ---
    
    if config["heart_rate"] and hist["bpm"]:
        # Use np.array for robust plotting against Plotly's type checks
        fig.add_trace(go.Scatter(y=np.array(hist["bpm"]), mode='lines', name='Heart Rate', line=dict(color='red')))
        
    if config["breathing"] and hist["breathing"]:
        fig.add_trace(go.Scatter(y=np.array(hist["breathing"]), mode='lines', name='Breathing', line=dict(color='blue')))
        
    if config["movement"] and hist["movement"]:
        # Scale movement for visibility
        scaled_mv = np.array(hist["movement"]) * 5
        fig.add_trace(go.Scatter(y=scaled_mv, mode='lines', name='Movement (x5)', line=dict(color='orange', dash='dot')))

    fig.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=20, b=20), 
        yaxis_title="Sensor Values",
        xaxis_title="Time (Last 100 Readings)",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-refresh loop for Dashboard
    time.sleep(1)
    st.rerun()

# ==========================================
# 7. MAIN ROUTING
# ==========================================
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_page()
    else:
        # Logout button header
        c1, c2 = st.columns([8,1])
        with c2:
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.rerun()
        
        # Router
        if st.session_state["role"] == "Laptop Operator (Monitor)":
            laptop_dashboard_page()
        else:
            mobile_sensor_page()

if __name__ == "__main__":
    main()
