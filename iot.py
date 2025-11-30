import streamlit as st
import cv2
import numpy as np
import tempfile
import plotly.graph_objs as go
from scipy.signal import find_peaks, savgol_filter
import os

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(page_title="Clinical Breath Analyzer", layout="wide", page_icon="🫁")

st.markdown("""
<style>
    .main-header { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .stat-box { padding: 15px; border-radius: 8px; text-align: center; color: white; }
    .stat-ok { background-color: #28a745; }
    .stat-crit { background-color: #dc3545; }
    .stat-warn { background-color: #ffc107; color: black; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ACCURATE PROCESSING ENGINE
# ==========================================
def extract_breathing_signal(video_path):
    """
    Uses Optical Flow to track vertical chest movement with high precision.
    Returns: Signal array, FPS, Duration
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0: fps = 30 # Fallback
    
    # Read first frame to select ROI (Region of Interest)
    ret, prev_frame = cap.read()
    if not ret: return [], 0, 0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    
    # FOCUS AREA: Center of the chest (middle 40% of screen)
    # This ignores head movement and background noise
    roi_x, roi_y, roi_w, roi_h = int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)
    
    motion_signal = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🔍 Analyzing chest movement via Optical Flow...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Optical Flow (Dense) only in the ROI
        # This gives us flow_x (horizontal) and flow_y (vertical) movement for every pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], 
            gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # We only care about VERTICAL (Y) movement for breathing
        # Summing absolute vertical changes captures the expansion/contraction intensity
        vertical_motion = np.mean(np.abs(flow[..., 1]))
        motion_signal.append(vertical_motion)
        
        prev_gray = gray
        frame_count += 1
        
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
    cap.release()
    status_text.empty()
    progress_bar.empty()
    
    return np.array(motion_signal), fps, total_frames / fps

def process_audio(video_path):
    """
    Extracts audio amplitude envelope to detect crying/gasps.
    """
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        
        if clip.audio is None:
            return None, 0
            
        # Get audio as numpy array
        # Read first 30 seconds max to save memory
        duration = min(clip.duration, 30) 
        audio = clip.audio.subclip(0, duration)
        sps = audio.fps # Samples per second
        data = audio.to_soundarray(fps=sps)
        
        # Convert stereo to mono
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        # Calculate Amplitude Envelope (simplify graph)
        window = int(sps * 0.1) # 100ms window
        envelope = [np.max(np.abs(data[i:i+window])) for i in range(0, len(data), window)]
        
        return np.array(envelope), sps/window # Return signal & new effective sampling rate
        
    except Exception as e:
        # Fallback if moviepy fails or no audio
        return None, 0

def analyze_signal(signal_data, fps):
    """
    Applies filters and peak detection to count breaths accurately.
    """
    # 1. Smooth the noisy optical flow signal (Savgol filter)
    # Window length must be odd and approx 1 second long
    window = int(fps) 
    if window % 2 == 0: window += 1
    if len(signal_data) < window: return 0, signal_data, []
    
    smooth_signal = savgol_filter(signal_data, window, 3)
    
    # 2. Normalize signal (0 to 1) for better plotting
    smooth_signal = (smooth_signal - np.min(smooth_signal)) / (np.max(smooth_signal) - np.min(smooth_signal) + 1e-6)
    
    # 3. Find Peaks (Breaths)
    # Min distance: 0.5 sec (nobody breathes faster than 120 bpm)
    # Prominence: ignores small jitters
    peaks, _ = find_peaks(smooth_signal, distance=fps*0.5, prominence=0.1)
    
    # 4. Calculate Rate (Breaths Per Minute)
    # We use the duration of the signal to get an accurate average
    duration_sec = len(signal_data) / fps
    bpm = (len(peaks) / duration_sec) * 60
    
    return bpm, smooth_signal, peaks

# ==========================================
# 3. MAIN APP INTERFACE
# ==========================================
def main():
    st.title("🫁 Neonatal Respiratory Analyzer")
    st.write("Upload a video of the baby. The system uses **Optical Flow Computer Vision** to track chest expansion and specific Audio Analysis for distress.")
    
    uploaded_file = st.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Save file locally for processing
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # --- LAYOUT ---
        col_video, col_analysis = st.columns([1, 1.5])
        
        # 1. Show Original Video
        with col_video:
            st.subheader("Original Footage")
            st.video(video_path)
            
        # 2. Run Analysis
        raw_signal, fps, duration = extract_breathing_signal(video_path)
        audio_env, audio_rate = process_audio(video_path)
        
        if len(raw_signal) > 0:
            bpm, smooth_signal, peaks = analyze_signal(raw_signal, fps)
            
            # --- RESULTS SECTION ---
            with col_analysis:
                st.subheader("Clinical Analysis Results")
                
                # DIAGNOSIS LOGIC
                status = "Normal Breathing"
                color = "stat-ok"
                msg = f"Breathing rate is within the healthy neonatal range ({int(bpm)} BPM)."
                
                if bpm > 60:
                    status = "Tachypnea (Respiratory Distress)"
                    color = "stat-crit"
                    msg = "⚠️ **High Alert:** Breathing is dangerously fast (>60 BPM). This is a primary sign of Pneumonia or RDS."
                elif bpm < 30:
                    status = "Bradypnea (Slow Breathing)"
                    color = "stat-warn"
                    msg = "⚠️ **Warning:** Breathing is slower than normal (<30 BPM). Monitor closely."

                # Status Box
                st.markdown(f"""
                <div class="stat-box {color}">
                    <h2 style="margin:0">{status}</h2>
                    <h1 style="font-size: 48px; margin:0">{int(bpm)}</h1>
                    <p>Breaths Per Minute</p>
                </div>
                <p style="margin-top:10px; padding:10px; background-color:#262730; border-radius:5px;">{msg}</p>
                """, unsafe_allow_html=True)
                
                # --- GRAPHS ---
                st.subheader("Physiological Signals")
                
                # Create Time Axis
                time_axis = np.linspace(0, duration, len(smooth_signal))
                
                fig = go.Figure()
                
                # Trace 1: Breathing Motion (Smoothed)
                fig.add_trace(go.Scatter(
                    x=time_axis, y=smooth_signal,
                    mode='lines', name='Chest Motion (Breathing)',
                    line=dict(color='#00CC96', width=3)
                ))
                
                # Trace 2: Detected Breaths (Peaks)
                peak_times = time_axis[peaks]
                peak_vals = smooth_signal[peaks]
                fig.add_trace(go.Scatter(
                    x=peak_times, y=peak_vals,
                    mode='markers', name='Detected Breath',
                    marker=dict(color='white', size=8, symbol='circle-open', line=dict(width=2))
                ))

                # Trace 3: Audio (if exists)
                if audio_env is not None and len(audio_env) > 0:
                    # Align audio time axis
                    audio_time = np.linspace(0, duration, len(audio_env))
                    # Normalize audio for visualization overlay
                    norm_audio = audio_env / (np.max(audio_env) + 1e-6) * 0.5 
                    
                    fig.add_trace(go.Scatter(
                        x=audio_time, y=norm_audio,
                        mode='lines', name='Audio Intensity (Cry/Gasp)',
                        line=dict(color='#EF553B', width=1),
                        opacity=0.6
                    ))

                fig.update_layout(
                    title="Breathing Pattern & Audio Correlation",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Normalized Intensity",
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- AUDIO DISTRESS CHECK ---
                if audio_env is not None:
                    avg_vol = np.mean(audio_env)
                    peak_vol = np.max(audio_env)
                    # Simple heuristic: if peak is 5x average, likely a cry or cough
                    if peak_vol > avg_vol * 5:
                        st.warning("🔊 **Audio Alert:** Sudden loud sounds detected (Crying, Coughing, or Gasping).")
                    else:
                        st.success("🔊 **Audio Status:** Audio levels are stable (Quiet breathing).")

        else:
            st.error("Could not process video. Please ensure the file is a valid video format.")
            
        # Cleanup
        os.unlink(tfile.name)

if __name__ == "__main__":
    main()
