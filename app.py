
import cv2
import streamlit as st
from datetime import datetime
from trackx_engine import TrackXEngine

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="TrackX Performance Dashboard",
    page_icon="🎯",
    layout="wide"
)

# ---------------- SESSION STATE ----------------

defaults = {
    "engine": None,
    "camera": None,
    "running": False,
    "events": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- HEADER ----------------

st.title("🎯 TrackX Performance Dashboard")
st.caption("AI-Powered Person Detection & Re-Identification")

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ TrackX Controls")

if st.sidebar.button(
    "▶ Start Monitoring",
    disabled=st.session_state.running
):
    try:
        with st.spinner("Loading TrackX engine..."):

            if st.session_state.engine is None:
                st.session_state.engine = TrackXEngine()

            camera = cv2.VideoCapture(0)

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if camera.isOpened():
                st.session_state.camera = camera
                st.session_state.running = True
                st.session_state.events = []
                st.rerun()
            else:
                camera.release()
                st.error("Unable to access webcam.")

    except Exception as e:
        st.error(f"Startup error: {e}")


if st.sidebar.button(
    "⏹ Stop Monitoring",
    disabled=not st.session_state.running
):

    st.session_state.running = False

    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

    st.rerun()

st.sidebar.divider()
st.sidebar.caption("TrackX | Performance Optimisation 2.0")

# ---------------- PERFORMANCE STATISTICS ----------------

st.subheader("📊 Live Performance Statistics")

col1, col2, col3 = st.columns(3)

fps_metric = col1.empty()
latency_metric = col2.empty()
yolo_metric = col3.empty()

col4, col5, col6 = st.columns(3)

reid_metric = col4.empty()
unique_metric = col5.empty()
active_metric = col6.empty()

st.divider()

# ---------------- MAIN LAYOUT ----------------

left, right = st.columns([2, 1])

with left:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with right:
    st.subheader("👥 Monitoring Activity")
    activity_placeholder = st.empty()

# ---------------- LIVE MONITORING ----------------

@st.fragment(run_every=0.2)
def live_monitor():

    if not st.session_state.running:

        fps_metric.metric("Processing FPS", "0.0")
        latency_metric.metric("Total Latency", "0 ms")
        yolo_metric.metric("YOLO Time", "0 ms")

        reid_metric.metric("ReID Stage", "0 ms")
        unique_metric.metric("Unique People", 0)
        active_metric.metric("Active People", 0)

        video_placeholder.info(
            "Click Start Monitoring to activate your webcam."
        )

        activity_placeholder.info(
            "Monitoring has not started."
        )

        return

    camera = st.session_state.camera
    engine = st.session_state.engine

    if camera is None or engine is None:
        st.error("Camera or engine is unavailable.")
        return

    success, frame = camera.read()

    if not success:
        st.error("Unable to read webcam frame.")
        return

    try:

        # Process each frame only once
        display, unique_people, active_people = (
            engine.process_frame(frame)
        )

        # ---------------- PERFORMANCE METRICS ----------------

        fps_metric.metric(
            "Processing FPS",
            f"{engine.last_fps:.2f}"
        )

        latency_metric.metric(
            "Total Latency",
            f"{engine.last_processing_ms:.1f} ms"
        )

        yolo_metric.metric(
            "YOLO Detection + Tracking",
            f"{engine.yolo_time_ms:.1f} ms"
        )

        reid_metric.metric(
            "ReID / Identity Stage",
            f"{engine.reid_time_ms:.1f} ms"
        )

        unique_metric.metric(
            "Unique People",
            unique_people
        )

        active_metric.metric(
            "Active People",
            active_people
        )

        # ---------------- CAMERA DISPLAY ----------------

        display = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            display,
            channels="RGB",
            width="stretch"
        )

        # ---------------- ACTIVITY PANEL ----------------

        activity_placeholder.markdown(
            f"""
            **Status:** 🟢 Monitoring

            **Last update:** {datetime.now().strftime("%H:%M:%S")}

            **Total identities:** {unique_people}

            **People currently visible:** {active_people}

            **Processing speed:** {engine.last_fps:.2f} FPS

            **YOLO processing:** {engine.yolo_time_ms:.1f} ms

            **ReID / identity stage:** {engine.reid_time_ms:.1f} ms
            """
        )

    except Exception as e:
        st.error(f"Processing error: {e}")


live_monitor()

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "TrackX | Performance Optimisation 2.0 | Session Identity Memory"
)
