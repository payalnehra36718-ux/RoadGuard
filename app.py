
import streamlit as st
import cv2
import pandas as pd
import tempfile
import os
from PIL import Image

st.set_page_config(
    page_title="RoadGuard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 RoadGuard")
st.subheader("Detect • Track • Prioritize")
st.write("AI-assisted road damage monitoring prototype")

st.info(
    "Demo version: extracts frames from road videos. "
    "Real pothole detection requires a trained AI model."
)

st.divider()

st.header("1. Upload Road Video")

uploaded = st.file_uploader(
    "Choose a road video",
    type=["mp4", "mov", "avi"]
)

if uploaded:
    st.video(uploaded)

    if st.button("Extract Road Frames"):
        temp_path = None
        cap = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as f:
                f.write(uploaded.getvalue())
                temp_path = f.name

            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                st.error("Could not read this video. Try an MP4 file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if not fps or fps <= 0:
                    fps = 30

                st.write(f"Video frames: {total}")
                st.write(f"Video duration: {total / fps:.1f} seconds")

                frames = []
                interval = max(1, int(fps * 2))
                index = 0

                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if index % interval == 0:
                        frame = cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        )
                        frames.append((index / fps, frame))

                    index += 1

                st.success(f"Extracted {len(frames)} frames.")

                for start in range(0, len(frames), 3):
                    cols = st.columns(3)
                    for col, item in zip(
                        cols, frames[start:start + 3]
                    ):
                        seconds, image = item
                        with col:
                            st.image(
                                image,
                                caption=f"Frame at {seconds:.1f}s",
                                use_container_width=True
                            )

        except Exception as e:
            st.error(f"Video processing error: {e}")

        finally:
            if cap is not None:
                cap.release()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()

st.header("2. Road Survey Dashboard")

st.caption(
    "The following records are illustrative demo data, "
    "not actual AI detections or live GPS reports."
)

data = pd.DataFrame([
    {
        "Defect ID": "RG001",
        "Road": "Road A",
        "Damage": "Pothole",
        "Severity": "High",
        "Observations": 3,
        "Priority": 90
    },
    {
        "Defect ID": "RG002",
        "Road": "Road B",
        "Damage": "Crack",
        "Severity": "Medium",
        "Observations": 2,
        "Priority": 60
    },
    {
        "Defect ID": "RG003",
        "Road": "Road C",
        "Damage": "Pothole",
        "Severity": "Low",
        "Observations": 1,
        "Priority": 25
    }
])

c1, c2, c3 = st.columns(3)
c1.metric("Demo Defects", len(data))
c2.metric(
    "High-Priority Defects",
    int((data["Severity"] == "High").sum())
)
c3.metric("Roads Listed", data["Road"].nunique())

st.dataframe(data, use_container_width=True)

st.subheader("3. Recommended Survey Order")

ranked = data.sort_values("Priority", ascending=False)
st.dataframe(
    ranked[["Road", "Damage", "Severity", "Priority"]],
    use_container_width=True
)

st.success(
    "RoadGuard demo is ready to test. "
    "Next development step: connect a trained road-damage model and GPS."
)
