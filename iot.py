import streamlit as st
import cv2
import numpy as np
import threading
import time
import collections
import plotly.graph_objs as go
import tempfile
import sys
from scipy.signal import find_peaks

# ==========================================
# 0. DEPENDENCY CHECK
# ==========================================
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
except ImportError:
    st.error("🚨 Critical Error: Missing dependencies. Please install: av, streamlit-webrtc, plotly, scipy")
    st.stop()

# ==========================================
# 1. CONFIGURATION: 4K SUPPORT
# ==========================================
st.set_page_config(page_title="Neonatal AI Guard", layout="wide", page_icon="👶")

# CSS for nice UI
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .status-card { padding: 15px; border-radius: 10px; color: white; margin-bottom: 10px; }
    .status-ok { background-color: #28a745; }
    .status-warn { background-color: #ffc107; color: black; }
    .status-crit { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# 4K / HD Constraints for Mobile Camera
MEDIA_CONSTRAINTS = {
    "video": {
        "width": {"min": 1280, "ideal": 3840, "max": 3840}, # Request 4K
        "height": {"min": 720, "ideal": 2160, "max": 2160},
        "frameRate": {"min": 24, "ideal": 30, "max": 60}, 
    },
    "audio": True
}

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
        self.data = {
            "live_vitals": {"bpm": 0, "rr": 0, "cry": False, "status": "Waiting..."},
            "latest_frame": None, # Stores the live video frame
            "history": {
                "bpm": collections.deque(maxlen=100),
                "rr": collections.deque(maxlen=100)
            }
        }

    def update(self, vitals, frame):
        with self._lock:
            self.data["live_vitals"] = vitals
            # We resize frame slightly for storage efficiency if it's true 4K, 
            # otherwise network lag will be massive. Keeping it HD (1080p) for display.
            if frame is not None:
                self.data["latest_frame"] = cv2.resize(frame, (1280, 720)) 
            
            # Update history
            if vitals["bpm"] > 0: self.data["history"]["bpm"].append(vitals["bpm"])
            if vitals["rr"] > 0: self.data["history"]["rr"].append(vitals["rr"])

    def get(self):
        with self._lock:
            return self.data.copy()

db = SharedDataManager()

# ==========================================
# 3. ADVANCED SIGNAL PROCESSING (FFT)
# ==========================================
class VitalSignProcessor:
    def __init__(self):
        self.green_buf = collections.deque(maxlen=300)
        self.breath_buf = collections.deque(maxlen=300)
        self.fps_est = 30
        self.last_t = time.time()

    def process(self, frame):
        # FPS Calculation
        curr_t = time.time()
        dt = curr_t - self.last_t
        if dt > 0: self.fps_est = 0.9 * self.fps_est + 0.1 * (1/dt)
        self.last_t = curr_t

        h, w, _ = frame.shape
        
        # 1. Breathing (Chest ROI)
        # Center-Bottom Region
        y1, y2 = int(h*0.4), int(h*0.8)
        x1, x2 = int(w*0.3), int(w*0.7)
        roi_breath = frame[y1:y2, x1:x2]
        gray_breath = cv2.cvtColor(roi_breath, cv2.COLOR_BGR2GRAY)
        breath_val = np.mean(gray_breath)
        self.breath_buf.append(breath_val)

        # 2. Heart Rate (Forehead ROI)
        # Top-Center Region
        fy1, fy2 = int(h*0.1), int(h*0.3)
        fx1, fx2 = int(w*0.4), int(w*0.6)
        roi_face = frame[fy1:fy2, fx1:fx2]
        
        if roi_face.size > 0:
            g_mean = np.mean(roi_face[:, :, 1])
            self.green_buf.append(g_mean)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2) # HR Box

        # Draw Breathing Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # RR Box

        # 3. Compute Rates
        hr = self._compute_fft(list(self.green_buf), 0.8, 3.0) # 48-180 bpm
        rr = self._compute_fft(list(self.breath_buf), 0.2, 1.0) # 12-60 bpm

        return frame, int(hr), int(rr)

    def _compute_fft(self, data, min_hz, max_hz):
        if len(data) < 60: return 0
        y = np.array(data)
        y = y - np.mean(y) # Detrend
        n = len(y)
        freqs = np.fft.rfftfreq(n, d=1/self.fps_est)
        mag = np.abs(np.fft.rfft(y))
        
        mask = (freqs >= min_hz) & (freqs <= max_hz)
        if not np.any(mask): return 0
        
        peak_idx = np.argmax(mag[mask])
        return freqs[mask][peak_idx] * 60

processor = VitalSignProcessor()

# ==========================================
# 4. WEBRTC CALLBACKS
# ==========================================
def video_frame_callback(frame: av.VideoFrame):
    try:
        img = frame.to_ndarray(format="bgr24")
    except: return frame

    # ANALYZE
    proc_img, hr, rr = processor.process(img)
    
    # PREDICT STATUS
    status = "Optimal"
    if rr > 60: status = "Tachypnea (High RR)"
    if hr > 180: status = "Tachycardia (High HR)"
    if hr < 90 and hr > 0: status = "Bradycardia (Low HR)"

    # UPDATE SHARED DB
    # We pass the processed image to the DB so the laptop sees the boxes/analysis
    current_audio = db.get()["live_vitals"]["cry"] # Keep audio state
    
    db.update({
        "bpm": hr,
        "rr": rr,
        "cry": current_audio,
        "status": status
    }, proc_img) # Send processed frame (with rectangles) to laptop

    return av.VideoFrame.from_ndarray(proc_img, format="bgr24")

def audio_frame_callback(frame: av.AudioFrame):
    try:
        sound = frame.to_ndarray()
        rms = np.sqrt(np.mean(sound**2))
        is_crying = rms > 2000
        
        # Update just the cry status, keep others same (not ideal in async but functional)
        data = db.get()
        vitals = data["live_vitals"]
        vitals["cry"] = is_crying
        db.update(vitals, data["latest_frame"])
        
    except: pass
    return frame

# ==========================================
# 5. UI PAGES
# ==========================================
def login_page():
    st.markdown("## 🔐 Secure Access")
    with st.form("login"):
        u = st.text_input("User", "admin")
        p = st.text_input("Password", type="password", value="admin")
        if st.form_submit_button("Login"):
            if u=="admin" and p=="admin":
                st.session_state["auth"] = True
                st.session_state["role"] = "Select"
                st.rerun()
            else:
                st.error("Access Denied")

def mobile_page():
    st.title("📱 Mobile Sensor Unit (4K Ready)")
    st.info("Ensure permissions are granted. Point camera at neonate.")
    
    webrtc_streamer(
        key="mobile_sender",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=MEDIA_CONSTRAINTS,
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        async_processing=True
    )
    
    st.write("Streaming active... check Laptop Dashboard.")

def laptop_page():
    st.title("💻 ICU Dashboard")
    
    tab1, tab2 = st.tabs(["📡 Live Mobile Feed", "📂 Upload Analysis"])
    
    # --- TAB 1: LIVE FEED FROM MOBILE ---
    with tab1:
        col_vid, col_stats = st.columns([2, 1])
        
        # Auto-refresh loop for real-time feel
        data = db.get()
        frame = data["latest_frame"]
        vitals = data["live_vitals"]
        
        with col_vid:
            if frame is not None:
                # Convert BGR to RGB
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            else:
                st.image(np.zeros((720, 1280, 3)), caption="Waiting for Mobile Stream...", use_container_width=True)
        
        with col_stats:
            st.subheader("Live Vitals")
            st.metric("Heart Rate", f"{vitals['bpm']} BPM")
            st.metric("Breathing Rate", f"{vitals['rr']} /min")
            
            # Status Logic
            s_color = "status-ok"
            if "High" in vitals['status'] or "Low" in vitals['status']: s_color = "status-crit"
            elif vitals['cry']: s_color = "status-warn"
            
            st.markdown(f"""
            <div class="status-card {s_color}">
                <h3>Status: {vitals['status']}</h3>
                <p>Crying: {'YES' if vitals['cry'] else 'NO'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Charts
            hist = data["history"]
            fig = go.Figure()
            if len(hist["bpm"]) > 0:
                fig.add_trace(go.Scatter(y=list(hist["bpm"]), mode='lines', name='HR', line=dict(color='red')))
            if len(hist["rr"]) > 0:
                fig.add_trace(go.Scatter(y=list(hist["rr"]), mode='lines', name='RR', line=dict(color='cyan')))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Refresh Feed"):
                st.rerun()

    # --- TAB 2: UPLOAD ANALYSIS ---
    with tab2:
        st.write("Upload a video (e.g., YouTube recording) for instant analysis.")
        f = st.file_uploader("Upload MP4", type=["mp4", "mov"])
        if f:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            cap = cv2.VideoCapture(tfile.name)
            
            st_frame = st.empty()
            st_metrics = st.empty()
            
            local_proc = VitalSignProcessor() # Separate processor for file
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Resize for processing speed
                frame = cv2.resize(frame, (1280, 720))
                p_frame, hr, rr = local_proc.process(frame)
                
                # Predict
                stat = "Optimal"
                if rr > 60: stat = "Respiratory Distress (Tachypnea)"
                
                st_frame.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                st_metrics.markdown(f"""
                ### File Analysis
                **HR:** {hr} BPM | **RR:** {rr} /min
                **Prediction:** {stat}
                """)
                
                # Simulate real-time playback
                # time.sleep(0.03)
            cap.release()

# ==========================================
# 6. MAIN ROUTING
# ==========================================
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login_page()
else:
    # Role Selection (Persists after login)
    if st.session_state.get("role") == "Select":
        st.title("Select Device Mode")
        c1, c2 = st.columns(2)
        if c1.button("📱 Mobile Sensor (Baby Unit)"):
            st.session_state["role"] = "Mobile"
            st.rerun()
        if c2.button("💻 Laptop Dashboard (Monitor)"):
            st.session_state["role"] = "Laptop"
            st.rerun()
    
    elif st.session_state["role"] == "Mobile":
        if st.sidebar.button("Logout / Switch"):
            st.session_state["auth"] = False
            st.rerun()
        mobile_page()
        
    elif st.session_state["role"] == "Laptop":
        if st.sidebar.button("Logout / Switch"):
            st.session_state["auth"] = False
            st.rerun()
        laptop_page()
        # Auto-rerun for laptop to keep fetching frames
        time.sleep(0.5)
        st.rerun()
