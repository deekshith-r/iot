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
    st.title("🚨 Missing Dependencies")
    st.error("The app requires 'av' and 'streamlit-webrtc' to run.")
    st.code("pip install av streamlit-webrtc", language="bash")
    try:
        st.stop()
    except Exception:
        sys.exit(1)

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

# RTC Configuration (Essential for camera/mic access over the web)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
            # Update live readings
            self.data_store["live_readings"] = readings
            
            # Update history only if the sensor is enabled (non-zero reading)
            if readings["bpm"] > 0:
                self.data_store["history"]["bpm"].append(readings["bpm"])
            if readings["breathing_rate"] > 0:
                self.data_store["history"]["breathing"].append(readings["breathing_rate"])
            if readings["movement_level"] > 0:
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
        # rPPG Buffer: Green channel averages (5 seconds @ ~30 FPS)
        self.green_buffer = collections.deque(maxlen=150) 
        # Optical Flow for breathing
        self.prev_points = None
        self.prev_gray = None
        self.frame_count = 0
    
    def process_heart_rate(self, frame):
        # 1. Select Region of Interest (ROI) for skin
        h, w, _ = frame.shape
        roi = frame[h//2-50:h//2+50, w//2-50:w//2+50]
        
        if roi.size == 0: return 0
        
        # 2. Average the green channel (most sensitive to blood flow)
        g_mean = np.mean(roi[:, :, 1])
        self.green_buffer.append(g_mean)
        
        # 3. Simple rPPG simulation: requires enough data (e.g., 5 seconds)
        if len(self.green_buffer) > 100:
            # Detrend the signal (subtract a mean baseline)
            detrended_signal = self.green_buffer - np.mean(self.green_buffer)
            
            # Simple simulation: BPM is inversely proportional to signal stability 
            # and proportional to variance over a window
            variance = np.var(detrended_signal)
            
            # Scale variance to a reasonable BPM range (90-160 for neonates)
            # This is a very rough simulation; real rPPG uses FFT
            simulated_bpm = int(90 + (variance * 1000) % 70) 
            
            return max(90, min(simulated_bpm, 180)) # Clamp to a realistic range
        return 0

    def process_breathing(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Define a small ROI around the chest/abdomen for breathing movement
        roi_x, roi_y = w//2 - 20, h//2 - 40
        roi_w, roi_h = 40, 80
        roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Use optical flow to track vertical movement (breathing)
        if self.prev_gray is not None and roi.size > 0:
            # Find corners to track
            p0 = cv2.goodFeaturesToTrack(roi, mask = None, maxCorners = 20, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
            
            if p0 is not None:
                if self.prev_points is not None:
                    # Calculate optical flow
                    p1, st, err = cv2.calcKLT(self.prev_gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], roi, self.prev_points, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    
                    if p1 is not None and st is not None:
                        # Select only successful points
                        good_new = p1[st==1]
                        good_old = self.prev_points[st==1]
                        
                        # Calculate vertical displacement (Y-axis)
                        dy = np.mean(good_new[:, 1] - good_old[:, 1]) if len(good_new) > 0 else 0
                        
                        # Simulate breathing rate: average movement magnitude
                        # Actual breathing rate requires FFT on the displacement signal
                        
                        # Store points for next frame
                        self.prev_points = good_new.reshape(-1, 1, 2)
                        
                        # Simulate breathing rate (15-40 breaths/min for neonate)
                        # Scale displacement into a rate:
                        simulated_rate = int(20 + abs(dy) * 5)
                        return max(15, min(simulated_rate, 40))

                # Initialize tracking points
                self.prev_points = p0
        
        self.prev_gray = roi.copy()
        return 0

    def process_movement(self, frame):
        # Uses frame differencing across the entire frame for gross motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        movement_score = 0
        if self.prev_gray is not None:
            # Use the full frame for motion detection here
            delta_frame = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Movement score is the count of significant pixel changes
            movement_score = np.sum(thresh) / 10000 
            
        self.prev_gray = gray
        return min(movement_score, 100)

# Global processor instance for continuity
processor = MediaProcessor() 

# WebRTC Video Callback
def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    config = state_manager.get_config()
    
    bpm, movement, breathing = 0, 0, 0

    if config["is_active"]:
        # 1. Run Analysis
        if config["heart_rate"]:
            bpm = processor.process_heart_rate(img)
            # Annotate ROI for rPPG
            h, w, _ = img.shape
            cv2.rectangle(img, (w//2-50, h//2-50), (w//2+50, h//2+50), (0, 255, 0), 2)
            
        if config["movement"]:
            movement = processor.process_movement(img)
            
        if config["breathing"]:
            breathing = processor.process_breathing(img)

        # 2. Update Shared State
        current_readings = state_manager.get_data()["live"]
        new_readings = {
            "bpm": bpm if config["heart_rate"] else 0,
            "breathing_rate": breathing if config["breathing"] else 0,
            "movement_level": movement if config["movement"] else 0,
            "is_crying": current_readings["is_crying"], # Audio state maintained
            "timestamp": time.time()
        }
        state_manager.update_readings(new_readings)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Audio Callback
def audio_frame_callback(frame: av.AudioFrame):
    sound = frame.to_ndarray()
    config = state_manager.get_config()
    
    # Placeholder for simple cry detection
    is_crying = False
    
    if config["is_active"] and config["cry_detection"]:
        rms = np.sqrt(np.mean(sound**2))
        # Simple threshold for loud noise/cry
        if rms > 1500: 
            is_crying = True
            
        # Update the cry status in the shared state
        current_readings = state_manager.get_data()["live"]
        current_readings["is_crying"] = is_crying
        current_readings["timestamp"] = time.time() # Ensure time updates
        state_manager.update_readings(current_readings)
        
    return frame # Return the frame unchanged

# ==========================================
# 4. PAGE: LOGIN & ROUTING
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
    st.info("Place this device near the neonate. Requires camera and microphone access.")
    
    config = state_manager.get_config()
    
    st.write("#### Active Requirements:")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HR", "ON" if config["heart_rate"] else "OFF")
    c2.metric("Breath", "ON" if config["breathing"] else "OFF")
    c3.metric("Motion", "ON" if config["movement"] else "OFF")
    c4.metric("Cry", "ON" if config["cry_detection"] else "OFF")

    # WebRTC Streamer
    if config["is_active"]:
        st.success("✅ Analysis Active - Streaming Data...")
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
        # Automatically re-run to check config status from the Laptop Operator
        time.sleep(2)
        st.rerun()

# ==========================================
# 6. PAGE: LAPTOP OPERATOR (DASHBOARD)
# ==========================================
def laptop_dashboard_page():
    st.markdown(f"### 💻 Monitoring Dashboard | Logged in as: {st.session_state['user']}")
    
    # 1. Sidebar Configuration
    with st.sidebar:
        st.header("Sensor Configuration")
        # Ensure checkboxes reflect current config, especially after a stop/start
        current_config = state_manager.get_config()
        hr_en = st.checkbox("Heart Rate (rPPG)", value=current_config.get("heart_rate", True))
        br_en = st.checkbox("Breathing Rate (Motion)", value=current_config.get("breathing", True))
        mv_en = st.checkbox("Movement (Optical Flow)", value=current_config.get("movement", True))
        cry_en = st.checkbox("Cry Detection (Audio)", value=current_config.get("cry_detection", True))
        
        st.divider()
        col_start, col_stop = st.columns(2)
        
        # START Button logic
        if col_start.button("▶ Start Analysis", type="primary"):
            state_manager.update_config({
                "heart_rate": hr_en,
                "breathing": br_en,
                "movement": mv_en,
                "cry_detection": cry_en,
                "is_active": True
            })
            st.toast("Analysis Started! Sensor unit must now connect.", icon="🚀")
            
        # STOP Button logic
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

    # 2. Status Logic & Alerts
    alerts = []
    
    bpm_status = "status-normal"
    if live["bpm"] > 0 and (live["bpm"] < 90 or live["bpm"] > 160):
        bpm_status = "status-critical"
        alerts.append(f"CRITICAL: Abnormal Heart Rate ({live['bpm']} BPM)")
    
    mv_status = "status-normal"
    if live["movement_level"] > 20: 
        mv_status = "status-warning"
        alerts.append("WARNING: High Movement Detected")
        
    cry_status = "status-normal"
    if live["is_crying"]:
        cry_status = "status-critical"
        alerts.append("🚨 ALERT: Crying Detected! Check on the neonate.")

    if alerts:
        for alert in alerts:
            st.error(alert)

    # 3. Live Metric Cards
    st.markdown("### Live Readings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # Only display if HR is enabled
        display_bpm = live['bpm'] if config["heart_rate"] else "N/A"
        st.markdown(f"""<div class="metric-card {bpm_status}"><h3>Heart Rate (BPM)</h3><h1>{display_bpm}</h1></div>""", unsafe_allow_html=True)
    with c2:
        # Only display if Breathing is enabled
        display_br = live['breathing_rate'] if config["breathing"] else "N/A"
        st.markdown(f"""<div class="metric-card status-normal"><h3>Breathing Rate</h3><h1>{display_br}</h1></div>""", unsafe_allow_html=True)
    with c3:
        # Only display if Movement is enabled
        display_mv = f"{int(live['movement_level'])}%" if config["movement"] else "N/A"
        st.markdown(f"""<div class="metric-card {mv_status}"><h3>Movement Level</h3><h1>{display_mv}</h1></div>""", unsafe_allow_html=True)
    with c4:
        # Only display if Cry Detection is enabled
        cry_text = 'CRYING' if live['is_crying'] else 'CALM'
        display_cry = cry_text if config["cry_detection"] else "N/A"
        st.markdown(f"""<div class="metric-card {cry_status}"><h3>Status</h3><h1>{display_cry}</h1></div>""", unsafe_allow_html=True)

    # 4. Charts
    st.markdown("### 📈 Live Trends (Last 100 updates)")
    fig = go.Figure()
    
    if config["heart_rate"]:
        fig.add_trace(go.Scatter(y=list(hist["bpm"]), mode='lines', name='Heart Rate (BPM)', line=dict(color='red')))
    if config["breathing"]:
        fig.add_trace(go.Scatter(y=list(hist["breathing"]), mode='lines', name='Breathing Rate', line=dict(color='blue')))
    if config["movement"]:
        # Scale movement for visibility on the same chart
        scaled_mv = [x * 5 for x in hist["movement"]]
        fig.add_trace(go.Scatter(y=scaled_mv, mode='lines', name='Movement (x5)', line=dict(color='orange', dash='dot')))

    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # 5. Continuous Refresh (Streamlit's way of real-time update)
    time.sleep(1) # Refresh every 1 second
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
        # Add Logout button for easy switching/exit
        c1, c2 = st.columns([8,1])
        with c2:
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.rerun()
        
        # Route to the appropriate page
        if st.session_state["role"] == "Laptop Operator (Monitor)":
            laptop_dashboard_page()
        else:
            mobile_sensor_page()

if __name__ == "__main__":
    main()
