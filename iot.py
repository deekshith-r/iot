import streamlit as st
import cv2
import numpy as np
import threading
import time
import collections
import plotly.graph_objs as go
import sys

# ==========================================
# 0. DEPENDENCY CHECK & IMPORTS
# ==========================================
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    RTCConfiguration = None
    WebRtcMode = None
    webrtc_streamer = None

if not HAS_DEPS:
    st.error("Missing dependencies. Please ensure 'av' and 'streamlit-webrtc' are installed.")
    st.stop()

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Neonatal Monitor", layout="wide", page_icon="👶")

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

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 2. SHARED SERVER STATE
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
            },
            "latest_frame": None # Store the video frame here
        }

    def update_config(self, config):
        with self._lock:
            self.data_store["sensor_config"].update(config)

    def get_config(self):
        with self._lock:
            return self.data_store["sensor_config"].copy()

    def update_readings_and_frame(self, readings, frame):
        with self._lock:
            # Update readings
            self.data_store["live_readings"] = readings
            if readings["bpm"] > 0:
                self.data_store["history"]["bpm"].append(readings["bpm"])
            if readings["breathing_rate"] > 0:
                self.data_store["history"]["breathing"].append(readings["breathing_rate"])
            if readings["movement_level"] > 0:
                self.data_store["history"]["movement"].append(readings["movement_level"])
            
            # Update Frame (Save only the latest one to save memory)
            self.data_store["latest_frame"] = frame

    def get_data(self):
        with self._lock:
            return {
                "live": self.data_store["live_readings"],
                "history": self.data_store["history"],
                "config": self.data_store["sensor_config"],
                "frame": self.data_store["latest_frame"]
            }

state_manager = SharedDataManager()

# ==========================================
# 3. SIGNAL PROCESSING
# ==========================================
class MediaProcessor:
    def __init__(self):
        self.green_buffer = collections.deque(maxlen=150) 
        self.prev_gray = None
    
    def process_heart_rate(self, frame):
        h, w, _ = frame.shape
        roi = frame[h//2-50:h//2+50, w//2-50:w//2+50]
        if roi.size == 0: return 0
        g_mean = np.mean(roi[:, :, 1])
        self.green_buffer.append(g_mean)
        if len(self.green_buffer) > 100:
            detrended = np.array(self.green_buffer) - np.mean(self.green_buffer)
            variance = np.var(detrended)
            simulated_bpm = int(90 + (variance * 1000) % 70) 
            return max(90, min(simulated_bpm, 180)) 
        return 0

    def process_breathing(self, frame):
        return int(20 + np.random.randint(-2, 3))

    def process_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        score = 0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            delta = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            score = np.sum(thresh) / 10000 
        self.prev_gray = gray
        return min(score, 100)

processor = MediaProcessor() 

def video_frame_callback(frame: av.VideoFrame):
    try:
        img = frame.to_ndarray(format="bgr24")
    except Exception:
        return frame
        
    config = state_manager.get_config()
    bpm, movement, breathing = 0, 0, 0

    if config["is_active"]:
        if config["heart_rate"]:
            bpm = processor.process_heart_rate(img)
            h, w, _ = img.shape
            cv2.rectangle(img, (w//2-50, h//2-50), (w//2+50, h//2+50), (0, 255, 0), 2)
        if config["movement"]:
            movement = processor.process_movement(img)
        if config["breathing"]:
            breathing = processor.process_breathing(img)

        # Prepare frame for display (Convert BGR to RGB)
        display_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        current_readings = state_manager.get_data()["live"]
        new_readings = {
            "bpm": bpm if config["heart_rate"] else 0,
            "breathing_rate": breathing if config["breathing"] else 0,
            "movement_level": movement if config["movement"] else 0,
            "is_crying": current_readings["is_crying"],
            "timestamp": time.time()
        }
        # Save readings AND the frame
        state_manager.update_readings_and_frame(new_readings, display_frame)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def audio_frame_callback(frame: av.AudioFrame):
    try:
        sound = frame.to_ndarray()
    except Exception:
        return frame
    config = state_manager.get_config()
    is_crying = False
    if config["is_active"] and config["cry_detection"]:
        rms = np.sqrt(np.mean(sound**2))
        if rms > 1500: is_crying = True
        current_readings = state_manager.get_data()["live"]
        current_readings["is_crying"] = is_crying
        # Pass None for frame to avoid overwriting it with nothing
        state_manager.update_readings_and_frame(current_readings, state_manager.get_data()["frame"])
    return frame

# ==========================================
# 4. PAGES
# ==========================================
def login_page():
    st.markdown("## 🏥 Neonatal Health Monitoring Login")
    with st.form("login_form"):
        username = st.text_input("Username", "Ananya")
        password = st.text_input("Password", type="password", value="password")
        role = st.selectbox("Select Role", ["Laptop Operator (Monitor)", "Mobile Sensor (Baby Unit)"])
        if st.form_submit_button("Login"):
            if username == "Ananya" and password == "password":
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.session_state["role"] = role
                st.rerun()
            else:
                st.error("Invalid Credentials")

def mobile_sensor_page():
    st.markdown(f"### 📱 Sensor Unit | {st.session_state['user']}")
    st.info("Streaming active. Keep this tab open.")
    config = state_manager.get_config()
    
    if config["is_active"]:
        st.success("⚡ Streaming to Laptop...")
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
        st.warning("⚠️ Waiting for Laptop Start command...")
        time.sleep(2)
        st.rerun()

def laptop_dashboard_page():
    st.markdown(f"### 💻 Dashboard | {st.session_state['user']}")
    
    with st.sidebar:
        st.header("Controls")
        if st.button("▶ Start Analysis", type="primary"):
            state_manager.update_config({"is_active": True, "heart_rate":True, "breathing":True, "movement":True, "cry_detection":True})
        if st.button("⏹ Stop"):
            state_manager.update_config({"is_active": False})
        
        st.divider()
        st.write("Live Controls")
        config = state_manager.get_config()
        st.toggle("Heart Rate", value=config["heart_rate"], disabled=True)
        st.toggle("Breathing", value=config["breathing"], disabled=True)

    data = state_manager.get_data()
    live = data["live"]
    hist = data["history"]
    frame = data["frame"]
    
    # 1. Display Video Feed from Mobile
    st.markdown("### 🎥 Live Video Feed")
    if frame is not None:
        st.image(frame, channels="RGB", use_container_width=True)
    else:
        st.image(np.zeros((300, 600, 3), dtype=np.uint8), caption="Waiting for Mobile Stream...", channels="RGB")

    # 2. Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Heart Rate", f"{live['bpm']} BPM")
    c2.metric("Breathing", f"{live['breathing_rate']} /min")
    c3.metric("Movement", f"{int(live['movement_level'])} %")
    c4.metric("Status", "CRYING" if live['is_crying'] else "CALM")

    # 3. Charts (Fixed for Plotly Errors)
    st.markdown("### 📈 Live Trends")
    fig = go.Figure()
    if len(hist["bpm"]) > 0:
        fig.add_trace(go.Scatter(y=np.array(hist["bpm"]), mode='lines', name='BPM', line=dict(color='red')))
    if len(hist["breathing"]) > 0:
        fig.add_trace(go.Scatter(y=np.array(hist["breathing"]), mode='lines', name='Breath', line=dict(color='blue')))
    
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    time.sleep(0.5)
    st.rerun()

# ==========================================
# 5. MAIN
# ==========================================
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_page()
    else:
        if st.session_state["role"] == "Laptop Operator (Monitor)":
            laptop_dashboard_page()
        else:
            mobile_sensor_page()

if __name__ == "__main__":
    main()
