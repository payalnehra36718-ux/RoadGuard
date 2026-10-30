
import os
import sqlite3
import math
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="RoadGuard",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ RoadGuard")
st.caption("AI-Based Road Damage Detection, Monitoring & Maintenance Prioritisation")


# ---------------- DATABASE ----------------

DB_PATH = "roadguard_history.db"
MODEL_PATH = "best.pt"


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            road_name TEXT,
            latitude REAL,
            longitude REAL,
            defect_class TEXT,
            confidence REAL,
            severity TEXT,
            priority REAL
        )
    """)
    conn.commit()
    return conn


def load_history():
    conn = connect_db()
    df = pd.read_sql_query(
        "SELECT * FROM observations ORDER BY id DESC",
        conn
    )
    conn.close()
    return df


# ---------------- MODEL ----------------

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)


# ---------------- LOCATION ----------------

st.sidebar.header("📍 Location")

st.sidebar.caption(
    "Allow browser location permission to retrieve your GPS coordinates."
)

if "latitude" not in st.session_state:
    st.session_state.latitude = 26.8500

if "longitude" not in st.session_state:
    st.session_state.longitude = 75.7600

if "road_name" not in st.session_state:
    st.session_state.road_name = ""


# Browser GPS component
location_html = """
<!DOCTYPE html>
<html>
<body style="font-family:Arial">
<button onclick="getLocation()" style="
    padding:10px 16px;
    background:#2563eb;
    color:white;
    border:0;
    border-radius:7px;
    cursor:pointer;">
    📍 Get My GPS Location
</button>

<p id="status" style="font-size:13px"></p>

<script>
function getLocation() {
    const status = document.getElementById("status");

    if (!navigator.geolocation) {
        status.innerText = "Geolocation is not supported.";
        return;
    }

    status.innerText = "Getting location...";

    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            status.innerText =
                "Latitude: " + lat.toFixed(6) +
                " | Longitude: " + lon.toFixed(6);

            window.parent.postMessage({
                type: "roadguard_location",
                latitude: lat,
                longitude: lon
            }, "*");
        },
        function(error) {
            status.innerText =
                "Location unavailable. Allow location permission or enter coordinates manually.";
        }
    );
}
</script>
</body>
</html>
"""

components.html(location_html, height=90)


# Manual coordinates remain available as a reliable fallback.
st.sidebar.text_input(
    "Road name",
    key="road_name",
    placeholder="Enter road name"
)

st.sidebar.number_input(
    "Latitude",
    format="%.6f",
    key="latitude"
)

st.sidebar.number_input(
    "Longitude",
    format="%.6f",
    key="longitude"
)

st.sidebar.caption(
    "Note: GPS button displays coordinates in the browser. "
    "The fields below are operator-entered unless updated manually."
)


# ---------------- SEVERITY AND PRIORITY ----------------

def suggested_severity(area):
    if area >= 100000:
        return "Critical"
    elif area >= 50000:
        return "Severe"
    elif area >= 15000:
        return "Moderate"
    return "Minor"


def priority_score(severity, confidence):
    severity_values = {
        "Minor": 1,
        "Moderate": 2,
        "Severe": 3,
        "Critical": 4
    }

    return round(
        severity_values.get(severity, 1) * confidence * 25,
        2
    )


def distance_m(lat1, lon1, lat2, lon2):
    R = 6371000

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Detect & Record",
    "📊 Priority Dashboard",
    "📁 Observation History"
])


# ---------------- TAB 1 ----------------

with tab1:

    st.header("Road Damage Detection")

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )

    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.10,
        step=0.05
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Road Image",
            use_container_width=True
        )

        if st.button("🚀 Analyse Image", type="primary"):

            model = get_model()

            if model is None:
                st.error("best.pt is missing. Add best.pt to your GitHub repository.")
            else:
                with st.spinner("Analysing road image..."):

                    results = model.predict(
                        image,
                        conf=confidence_threshold,
                        verbose=False
                    )

                    result = results[0]

                    st.subheader("Detection Result")

                    st.image(
                        result.plot(),
                        caption="AI Detection Output",
                        use_container_width=True
                    )

                    if result.boxes is None or len(result.boxes) == 0:
                        st.warning(
                            "No damage detected. Try a clear road-damage image "
                            "or a lower confidence threshold."
                        )

                    else:

                        rows = []

                        for box in result.boxes:

                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            area = max(0, x2 - x1) * max(0, y2 - y1)

                            severity = suggested_severity(area)

                            rows.append({
                                "Class ID": class_id,
                                "Confidence": round(confidence, 3),
                                "Bounding Box Area": round(area, 2),
                                "Suggested Severity": severity
                            })

                        detection_df = pd.DataFrame(rows)

                        st.dataframe(
                            detection_df,
                            use_container_width=True
                        )

                        st.info(
                            "Severity is currently a demo heuristic based on "
                            "bounding-box area, not a clinically or engineering "
                            "validated road-condition measurement."
                        )

                        st.subheader("Save Observation")

                        selected_index = st.selectbox(
                            "Select detected defect",
                            range(len(rows)),
                            format_func=lambda x: (
                                f"Detection {x + 1} — "
                                f"Class {rows[x]['Class ID']}"
                            )
                        )

                        selected = rows[selected_index]

                        severity = st.selectbox(
                            "Confirm severity",
                            ["Minor", "Moderate", "Severe", "Critical"],
                            index=[
                                "Minor", "Moderate", "Severe", "Critical"
                            ].index(selected["Suggested Severity"])
                        )

                        road_name = st.session_state.road_name.strip()
                        latitude = st.session_state.latitude
                        longitude = st.session_state.longitude

                        priority = priority_score(
                            severity,
                            selected["Confidence"]
                        )

                        st.metric("Maintenance Priority Score", priority)

                        if st.button("💾 Save Observation"):

                            if not road_name:
                                st.error("Please enter the road name.")
                            else:
                                conn = connect_db()

                                conn.execute("""
                                    INSERT INTO observations
                                    (date, road_name, latitude, longitude,
                                     defect_class, confidence, severity, priority)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    road_name,
                                    latitude,
                                    longitude,
                                    str(selected["Class ID"]),
                                    selected["Confidence"],
                                    severity,
                                    priority
                                ))

                                conn.commit()
                                conn.close()

                                st.success("Observation saved successfully!")


# ---------------- TAB 2 ----------------

with tab2:

    st.header("Maintenance Priority Dashboard")

    history = load_history()

    if history.empty:
        st.info("No observations recorded yet.")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Observations", len(history))
        col2.metric("Severe / Critical", len(
            history[history["severity"].isin(["Severe", "Critical"])]
        ))
        col3.metric(
            "Average Priority",
            round(history["priority"].mean(), 2)
        )

        st.subheader("Highest Priority Defects")

        dashboard = history.sort_values(
            "priority",
            ascending=False
        )

        st.dataframe(
            dashboard,
            use_container_width=True
        )

        st.bar_chart(
            dashboard.set_index("road_name")["priority"]
        )


# ---------------- TAB 3 ----------------

with tab3:

    st.header("Observation History")

    history = load_history()

    if history.empty:
        st.info("No saved observations yet.")

    else:
        st.dataframe(
            history,
            use_container_width=True
        )

        csv = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Observation History",
            data=csv,
            file_name="roadguard_observations.csv",
            mime="text/csv"
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "RoadGuard | AI-assisted road monitoring prototype | "
    "GPS and road details require verification."
)

