import os
import sqlite3
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="RoadGuard | Road Damage Monitor", page_icon="🛣️", layout="wide")

DB_PATH = "roadguard_history.db"
MODEL_PATH = "best.pt"

SEVERITY_POINTS = {"Minor": 1, "Moderate": 2, "Severe": 3, "Critical": 4}

def connect_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            defect_key TEXT,
            observed_at TEXT,
            road_name TEXT,
            latitude REAL,
            longitude REAL,
            class_name TEXT,
            confidence REAL,
            severity TEXT,
            bbox_area_pct REAL,
            status TEXT DEFAULT 'Open'
        )
    """)
    con.commit()
    return con

def distance_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2-lat1)
    dl = radians(lon2-lon1)
    a = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2*r*atan2(sqrt(a), sqrt(1-a))

def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

def suggested_severity(area_pct):
    # Demo heuristic only: bounding-box area is NOT a validated measure of pothole danger.
    if area_pct >= 20:
        return "Critical"
    if area_pct >= 10:
        return "Severe"
    if area_pct >= 3:
        return "Moderate"
    return "Minor"

def nearest_existing(con, lat, lon, radius=25):
    rows = con.execute(
        "SELECT defect_key, latitude, longitude FROM observations WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()
    for key, old_lat, old_lon in rows:
        if distance_m(lat, lon, old_lat, old_lon) <= radius:
            return key
    return None

def priority_score(severity, observation_count):
    # Proposed prototype score; weights require validation with road-authority data.
    return min(100, SEVERITY_POINTS[severity] * 15 + min(observation_count, 5) * 8)

st.title("🛣️ RoadGuard")
st.caption("AI-assisted road damage detection, defect history and maintenance prioritisation — prototype")

with st.sidebar:
    st.header("Project status")
    st.write("✅ YOLO image detection")
    st.write("🧪 Severity suggestion (demo heuristic)")
    st.write("🧪 GPS entered by operator")
    st.write("🧪 Repeat observations matched by nearby GPS")
    st.write("🧪 Priority ranking")
    st.write("🔜 Automated GPS capture / government integration")
    st.divider()
    st.caption("This prototype is a decision-support demonstration, not an engineering-grade road safety assessment.")

con = connect_db()

tab1, tab2, tab3 = st.tabs(["1. Detect & record", "2. Priority dashboard", "3. Observation history"])

with tab1:
    st.subheader("Upload a road image")
    st.info("The current model returns dataset class IDs. Confirm the class mapping before interpreting a class as a pothole.")
    uploaded = st.file_uploader("Choose a road image", type=["jpg", "jpeg", "png"])
    c1, c2, c3 = st.columns(3)
    with c1:
        road_name = st.text_input("Road / area name", placeholder="e.g. Mansarovar main road")
    with c2:
        latitude = st.number_input("Latitude", value=26.8500, format="%.6f")
    with c3:
        longitude = st.number_input("Longitude", value=75.7600, format="%.6f")

    confidence = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded road image", use_container_width=True)
        if st.button("Analyse image", type="primary"):
            model = get_model()
            if model is None:
                st.error("Model file best.pt was not found. Put best.pt in the same folder as app.py.")
            else:
                with st.spinner("Analysing image..."):
                    prediction = model.predict(image, conf=confidence, verbose=False)[0]
                    plotted = prediction.plot()
                    st.image(plotted, caption="YOLO detections", channels="BGR", use_container_width=True)
                    names = prediction.names
                    boxes = prediction.boxes
                    if boxes is None or len(boxes) == 0:
                        st.warning("No objects detected at this confidence threshold.")
                    else:
                        st.success(f"{len(boxes)} object(s) detected.")
                        rows = []
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                            area_pct = ((x2-x1)*(y2-y1)/(image.width*image.height))*100
                            suggested = suggested_severity(area_pct)
                            rows.append({
                                "Detection": i+1,
                                "Class ID": cls_id,
                                "Class label": str(names.get(cls_id, cls_id)),
                                "Confidence": round(conf, 3),
                                "Box area (%)": round(area_pct, 2),
                                "Suggested severity": suggested
                            })
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.warning("Severity is only a rough bounding-box-size suggestion. It is not a validated pothole-depth, risk or safety measurement.")
                        selected = st.selectbox(
                            "Choose the detection to record",
                            options=list(range(len(rows))),
                            format_func=lambda i: f"Detection {i+1} — class {rows[i]['Class ID']} ({rows[i]['Confidence']:.2f})"
                        )
                        suggested_value = rows[selected]["Suggested severity"]
                        severity = st.selectbox(
                            "Confirm severity (operator review)",
                            ["Minor", "Moderate", "Severe", "Critical"],
                            index=["Minor", "Moderate", "Severe", "Critical"].index(suggested_value)
                        )
                        if st.button("Save this observation"):
                            existing_key = nearest_existing(con, latitude, longitude)
                            defect_key = existing_key or f"RG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            count_before = con.execute(
                                "SELECT COUNT(*) FROM observations WHERE defect_key=?", (defect_key,)
                            ).fetchone()[0]
                            row = rows[selected]
                            con.execute("""
                                INSERT INTO observations
                                (defect_key, observed_at, road_name, latitude, longitude, class_name,
                                 confidence, severity, bbox_area_pct, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open')
                            """, (
                                defect_key, datetime.now().isoformat(timespec="seconds"),
                                road_name, latitude, longitude, row["Class label"],
                                row["Confidence"], severity, row["Box area (%)"]
                            ))
                            con.commit()
                            st.success(
                                f"Saved as {defect_key}. This is observation #{count_before+1} "
                                f"for the matched location."
                            )

with tab2:
    st.subheader("Which road defects need attention first?")
    df = pd.read_sql_query("SELECT * FROM observations", con)
    if df.empty:
        st.info("No observations recorded yet. Upload an image and save a detection in the first tab.")
    else:
        counts = df.groupby("defect_key").size().rename("Observation count")
        latest = df.sort_values("observed_at").groupby("defect_key").tail(1).copy()
        latest = latest.merge(counts, on="defect_key")
        latest["Priority score"] = latest.apply(
            lambda r: priority_score(r["severity"], int(r["Observation count"])), axis=1
        )
        latest["Priority"] = latest["Priority score"].apply(
            lambda x: "Critical" if x >= 75 else ("High" if x >= 55 else ("Medium" if x >= 35 else "Low"))
        )
        latest = latest.sort_values("Priority score", ascending=False)
        st.caption("Prototype ranking: severity + repeat-observation count. Score weights are proposed and need validation.")
        st.dataframe(
            latest[["defect_key", "road_name", "severity", "Observation count",
                    "Priority", "Priority score", "observed_at", "status"]],
            use_container_width=True, hide_index=True
        )
        st.metric("Unique tracked locations", latest["defect_key"].nunique())
        st.metric("Open observations", int((df["status"] == "Open").sum()))
        st.download_button(
            "Download priority report (CSV)",
            latest.to_csv(index=False).encode("utf-8"),
            file_name="roadguard_priority_report.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Repeated observations / defect history")
    history = pd.read_sql_query(
        "SELECT * FROM observations ORDER BY observed_at DESC", con
    )
    if history.empty:
        st.info("No saved observations yet.")
    else:
        selected_key = st.selectbox("Select a tracked defect", sorted(history["defect_key"].unique()))
        st.dataframe(
            history[history["defect_key"] == selected_key],
            use_container_width=True, hide_index=True
        )
        st.caption("Nearby GPS points within 25 metres are grouped as the same defect in this prototype. This matching radius needs field testing.")
        st.download_button(
            "Download full observation history (CSV)",
            history.to_csv(index=False).encode("utf-8"),
            file_name="roadguard_observation_history.csv",
            mime="text/csv"
        )

st.divider()
st.caption("RoadGuard prototype | Detection model: YOLOv8n | Human review required before maintenance decisions.")
