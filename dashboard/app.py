"""
Traffic-Eye AI — Analytics Dashboard with Live Detection

Streamlit dashboard with two modes:
  1. Live Detection — Upload a video, click "Start Detection", watch
     violations detected frame-by-frame in real time.
  2. Analytics — Browse historical evidence records from evidence_store/.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import base64
import html
import json
import os
import sys
import glob
import tempfile
import time
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Ensure project root is on sys.path
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DASHBOARD_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─── Configuration ────────────────────────────────────────────────────

# Frame-render throttle: show every Nth frame in the live view.
# Set to 1 to show every frame (slowest UI), 3 for smooth balance.
# WHY THIS EXISTS: Each st.image() call has Streamlit redraw overhead
# (~10-30ms). Rendering every frame at 30+ FPS causes the UI to lag.
# The pipeline still processes EVERY frame for violations regardless.
LIVE_DISPLAY_EVERY_N_FRAMES = 3

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic-Eye AI — Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #3d3d5c;
    }
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 107, 53, 0.3);
    }
    .live-violation-entry {
        background: rgba(255, 107, 53, 0.08);
        border-left: 3px solid #FF6B35;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .evidence-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
        line-height: 1.35;
        table-layout: fixed;
    }
    .evidence-table th,
    .evidence-table td {
        border: 1px solid rgba(250, 250, 250, 0.12);
        padding: 0.55rem 0.65rem;
        text-align: left;
        vertical-align: middle;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .evidence-table th {
        color: rgba(250, 250, 250, 0.68);
        background: rgba(250, 250, 250, 0.04);
        font-weight: 600;
    }
    .evidence-table a {
        color: #FF8C61;
        font-weight: 600;
        text-decoration: none;
    }
    .evidence-table a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading (for Analytics tab) ─────────────────────────────────
@st.cache_data(ttl=10)  # Refresh every 10 seconds
def load_evidence_records(evidence_dir='evidence_store'):
    """Load all evidence JSON records from the store."""
    records = []
    records_dir = os.path.join(evidence_dir, 'records')

    if not os.path.exists(records_dir):
        return pd.DataFrame()

    for filepath in glob.glob(os.path.join(records_dir, '*.json')):
        try:
            with open(filepath, 'r') as f:
                record = json.load(f)
                records.append(record)
        except (json.JSONDecodeError, IOError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date

    return df


def calculate_tci(df, time_window_hours=24):
    """
    Calculate Traffic Compliance Index per zone.

    Formula: TCI = 100 - violation_rate
    violation_rate = (violations_in_window / max_expected_violations) * 100
    max_expected_violations is normalized to 50 per zone per window (tunable).
    TCI is clamped to [0, 100].
    """
    if df.empty:
        return pd.DataFrame()

    MAX_EXPECTED = 50  # Expected max violations per zone per time window

    zone_violations = df.groupby('camera_id').size().reset_index(name='violation_count')
    zone_violations['violation_rate'] = (zone_violations['violation_count'] / MAX_EXPECTED) * 100
    zone_violations['violation_rate'] = zone_violations['violation_rate'].clip(0, 100)
    zone_violations['tci'] = (100 - zone_violations['violation_rate']).clip(0, 100)

    # Add location names
    location_map = {}
    config_path = os.path.join(_PROJECT_ROOT, 'config', 'camera_locations.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cameras = json.load(f).get('cameras', {})
            location_map = {k: v.get('location_name', k) for k, v in cameras.items()}

    zone_violations['location'] = zone_violations['camera_id'].map(
        lambda x: location_map.get(x, x)
    )

    return zone_violations


def resolve_evidence_image_path(image_path):
    """Resolve stored evidence image paths to local files for Streamlit."""
    if not image_path or pd.isna(image_path):
        return None

    image_path = str(image_path)
    candidate = image_path if os.path.isabs(image_path) else os.path.join(_PROJECT_ROOT, image_path)
    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
        return candidate

    return None


def format_record_timestamp(timestamp):
    """Format a record timestamp for compact display."""
    parsed = pd.to_datetime(timestamp, errors='coerce')
    if pd.isna(parsed):
        return "N/A"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def format_percent(value):
    """Format numeric confidence values without breaking on missing data."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "N/A"


def format_frame_number(value):
    """Format frame numbers without pandas converting them to floats."""
    try:
        if pd.isna(value):
            return "N/A"
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"


def image_open_link(image_path):
    """Build a browser-openable evidence image link."""
    resolved_path = resolve_evidence_image_path(image_path)
    if not resolved_path:
        return "No image"

    with open(resolved_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")

    filename = html.escape(os.path.basename(resolved_path))
    return (
        f'<a href="data:image/jpeg;base64,{encoded}" '
        f'target="_blank" title="{filename}">Open Image</a>'
    )


def render_evidence_table(records, empty_message):
    """Render violation/evidence records as a compact table with image links."""
    if not records:
        st.info(empty_message)
        return

    headers = [
        "Image", "Timestamp", "Frame", "Type", "Confidence",
        "Risk", "Status", "Plate", "Location",
    ]
    rows = []
    for record in records:
        row_values = [
            image_open_link(record.get("evidence_image_path")),
            html.escape(format_record_timestamp(record.get("timestamp"))),
            html.escape(format_frame_number(record.get("frame_number"))),
            html.escape(record.get("violation_type", "unknown").replace("_", " ").title()),
            html.escape(format_percent(record.get("confidence"))),
            html.escape(str(record.get("risk_category", "N/A"))),
            html.escape(str(record.get("status", "N/A")).replace("_", " ").title()),
            html.escape(str(record.get("vehicle_plate") or "N/A")),
            html.escape(str(record.get("camera_location", "N/A"))),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{value}</td>" for value in row_values)
            + "</tr>"
        )

    table = (
        '<table class="evidence-table">'
        "<thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    st.markdown(table, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# LIVE DETECTION TAB
# ══════════════════════════════════════════════════════════════════════

def render_live_detection_tab():
    """Render the Live Detection UI — upload, process, watch in real time."""

    st.markdown("###  Live Video Detection")
    st.markdown(
        "Upload a traffic video, click **Start Detection**, and watch "
        "violations detected frame-by-frame in real time."
    )

    # ── File uploader ─────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload traffic video",
        type=["mp4", "mov", "avi"],
        help="Supported formats: MP4, MOV, AVI",
    )

    if uploaded_file is None:
        st.info("👆 Upload a video file to get started.")
        return

    st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.1f} MB)")

    # ── Configuration row ─────────────────────────────────────────
    col_cam, col_max = st.columns(2)
    with col_cam:
        camera_id = st.text_input("Camera ID", value="CAM_001")
    with col_max:
        max_frames_input = st.number_input(
            "Max Frames (0 = all)", min_value=0, value=0, step=50,
            help="Limit processing to N frames (0 = process entire video)"
        )
    max_frames = max_frames_input if max_frames_input > 0 else None

    # ── Start button ──────────────────────────────────────────────
    if not st.button(" Start Detection", type="primary", use_container_width=True):
        return

    # ── Save uploaded file to temp path ───────────────────────────
    upload_dir = os.path.join(_PROJECT_ROOT, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_input_path = os.path.join(upload_dir, uploaded_file.name)
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Output video path
    output_dir = os.path.join(_PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, f"output_{uploaded_file.name}")

    # ── Create UI placeholders ────────────────────────────────────
    st.markdown("---")
    st.markdown("###  Live Processing")

    progress_bar = st.progress(0, text="Initializing pipeline...")

    # Live metrics row
    metric_cols = st.columns(4)
    metric_frames = metric_cols[0].empty()
    metric_violations = metric_cols[1].empty()
    metric_fps = metric_cols[2].empty()
    metric_elapsed = metric_cols[3].empty()

    # Live video frame display
    frame_display = st.empty()

    # Live violation log
    st.markdown("### 📋 Live Violation Log")
    violation_log_placeholder = st.empty()

    # ── Import and run the core processor ─────────────────────────
    from core.processor import process_video_stream

    all_violations_log: list[dict] = []
    last_update = None

    try:
        for update in process_video_stream(
            source_path=temp_input_path,
            output_path=out_video_path,
            camera_id=camera_id,
            config_dir=os.path.join(_PROJECT_ROOT, "config"),
            max_frames=max_frames,
        ):
            last_update = update
            frame_idx = update["frame_idx"]
            total = update["total_frames"]
            fps = update["fps_so_far"]
            elapsed = update["elapsed_seconds"]
            v_count = update["violation_count"]

            # Update progress bar
            if total > 0:
                pct = min((frame_idx + 1) / total, 1.0)
                progress_bar.progress(pct, text=f"Frame {frame_idx + 1}/{total}")

            # Update live metrics (every frame is fine — these are lightweight)
            metric_frames.metric("Frames", f"{frame_idx + 1}/{total}")
            metric_violations.metric("Violations", v_count)
            metric_fps.metric("FPS", f"{fps:.1f}")
            metric_elapsed.metric("Elapsed", f"{elapsed:.1f}s")

            # Update video display (throttled to every Nth frame)
            if frame_idx % LIVE_DISPLAY_EVERY_N_FRAMES == 0:
                # Convert BGR → RGB for Streamlit display
                rgb_frame = cv2.cvtColor(update["annotated_frame"], cv2.COLOR_BGR2RGB)
                frame_display.image(rgb_frame, channels="RGB", use_container_width=True)

            # Append new violations to log
            if update["new_violations"]:
                for rec in update["new_violations"]:
                    live_record = dict(rec)
                    live_record["frame_number"] = (
                        rec.get("frame_number")
                        if rec.get("frame_number") is not None
                        else frame_idx
                    )
                    all_violations_log.append(live_record)

                with violation_log_placeholder.container():
                    render_evidence_table(
                        list(reversed(all_violations_log[-10:])),
                        "No violations detected yet.",
                    )

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    # ── Completion summary ────────────────────────────────────────
    progress_bar.progress(1.0, text="✅ Processing complete!")

    if last_update is not None:
        elapsed = last_update["elapsed_seconds"]
        processed = last_update["frame_idx"] + 1
        avg_fps = processed / elapsed if elapsed > 0 else 0.0
        total_violations = last_update["violation_count"]

        st.markdown("---")
        st.markdown("### ✅ Processing Complete")

        summary_cols = st.columns(5)
        summary_cols[0].metric("Total Frames", processed)
        summary_cols[1].metric("Total Time", f"{elapsed:.1f}s")
        summary_cols[2].metric("Avg FPS", f"{avg_fps:.1f}")
        summary_cols[3].metric("Violations Found", total_violations)
        summary_cols[4].metric("Output Video", "Ready ✓")

        # Provide download link for output video
        if os.path.exists(out_video_path):
            with open(out_video_path, "rb") as video_file:
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=video_file.read(),
                    file_name=f"output_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True,
                )

        # Report suppressed exceptions
        suppressed = last_update.get("suppressed_counts", {})
        for exc_key, count in suppressed.items():
            if count > 0:
                st.warning(
                    f"⚠️ {exc_key[0]} raised {exc_key[1]} {count} more time(s) "
                    f"after first occurrence (suppressed)"
                )

    # Clear analytics cache so new evidence records show up immediately
    st.cache_data.clear()

    if not all_violations_log:
        st.info("ℹ️ No violations detected in this video. This is expected for synthetic/test footage.")


# ══════════════════════════════════════════════════════════════════════
# ANALYTICS TAB (existing functionality — unchanged)
# ══════════════════════════════════════════════════════════════════════

def render_analytics_tab():
    """Render the existing Analytics view — reads from evidence_store/."""

    # Sidebar filters
    evidence_dir = os.path.join(_PROJECT_ROOT, "evidence_store")

    # Load data
    df = load_evidence_records(evidence_dir)

    if df.empty:
        st.warning("⚠️ No evidence records found. Run the pipeline or use Live Detection to generate data.")
        st.info(
            "To generate data, run:\n"
            "```bash\n"
            "python pipeline.py --source data/test_videos/your_video.mp4 --output output/\n"
            "```\n"
            "Or use the **🎬 Live Detection** tab to upload and process a video directly."
        )
        return

    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Analytics Filters")

        violation_types = ['All'] + sorted(df['violation_type'].unique().tolist())
        selected_type = st.selectbox("Violation Type", violation_types)

        status_filter = st.selectbox("Status", ['All', 'formal_record', 'pending_review'])

        risk_filter = st.selectbox("Risk Category", ['All', 'High', 'Medium', 'Low'])

    # Apply filters
    filtered_df = df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['violation_type'] == selected_type]
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    if risk_filter != 'All':
        filtered_df = filtered_df[filtered_df['risk_category'] == risk_filter]

    # ─── Key Metrics Row ──────────────────────────────────────────
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Violations", len(filtered_df))
    with col2:
        formal = len(filtered_df[filtered_df['status'] == 'formal_record'])
        st.metric("Formal Records", formal)
    with col3:
        pending = len(filtered_df[filtered_df['status'] == 'pending_review'])
        st.metric("Pending Review", pending)
    with col4:
        high_risk = len(filtered_df[filtered_df['risk_category'] == 'High'])
        st.metric("High Risk", high_risk)
    with col5:
        avg_conf = filtered_df['confidence'].mean() if not filtered_df.empty else 0
        st.metric("Avg Confidence", f"{avg_conf:.0%}")

    st.markdown("---")

    # ─── Violation Counts by Type ─────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 Violations by Type")
        if not filtered_df.empty:
            type_counts = filtered_df['violation_type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#FF6B35', '#FF8C61', '#FFB088', '#FFD4B8',
                      '#4ECDC4', '#45B7AA', '#3DA190']
            bars = ax.bar(range(len(type_counts)), type_counts.values,
                         color=colors[:len(type_counts)])
            ax.set_xticks(range(len(type_counts)))
            ax.set_xticklabels([t.replace('_', '\n') for t in type_counts.index],
                              rotation=0, fontsize=9)
            ax.set_ylabel('Count')
            ax.set_title('Violation Distribution')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for bar, count in zip(bars, type_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                       str(count), ha='center', va='bottom', fontweight='bold')

            st.pyplot(fig)
            plt.close(fig)

    with col_right:
        st.markdown("### 🗺️ Violation Density by Camera/Zone")
        if not filtered_df.empty and 'camera_id' in filtered_df.columns:
            zone_data = filtered_df.groupby(['camera_id', 'camera_location']).agg(
                violation_count=('evidence_id', 'count'),
                avg_risk=('risk_score', 'mean')
            ).reset_index()

            fig, ax = plt.subplots(figsize=(8, 5))
            locations = zone_data['camera_location'].tolist()
            counts = zone_data['violation_count'].tolist()
            risk_colors = zone_data['avg_risk'].tolist()

            scatter = ax.barh(range(len(locations)), counts,
                             color=[plt.cm.RdYlGn_r(r) for r in risk_colors])
            ax.set_yticks(range(len(locations)))
            ax.set_yticklabels(locations, fontsize=9)
            ax.set_xlabel('Violation Count')
            ax.set_title('Violations by Location (color = avg risk)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for i, count in enumerate(counts):
                ax.text(count + 0.2, i, str(count), va='center', fontweight='bold')

            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # ─── Top 10 Junctions ─────────────────────────────────────────
    col_tci, col_top = st.columns(2)

    with col_top:
        st.markdown("###  Top 10 Junctions by Violation Count")
        if not filtered_df.empty and 'camera_location' in filtered_df.columns:
            top_junctions = (filtered_df['camera_location']
                           .value_counts()
                           .head(10)
                           .reset_index())
            top_junctions.columns = ['Junction', 'Violations']
            st.dataframe(top_junctions, use_container_width=True, hide_index=True)

    # ─── Traffic Compliance Index ─────────────────────────────────
    with col_tci:
        st.markdown("###  Traffic Compliance Index (TCI)")
        tci_df = calculate_tci(filtered_df)
        if not tci_df.empty:
            for _, row in tci_df.iterrows():
                tci_val = row['tci']
                color = '#2ecc71' if tci_val >= 80 else '#f39c12' if tci_val >= 50 else '#e74c3c'
                st.markdown(
                    f"**{row['location']}**: "
                    f"<span style='color:{color}; font-size:1.3em; font-weight:700'>"
                    f"{tci_val:.1f}</span> / 100",
                    unsafe_allow_html=True
                )
                st.progress(tci_val / 100)

            st.caption(
                "TCI = 100 − (violations / max_expected × 100). "
                "Max expected = 50 violations per zone per window. "
                "Higher is better."
            )

    st.markdown("---")

    # ─── Temporal Patterns ────────────────────────────────────────
    st.markdown("### 🕐 Temporal Patterns")
    col_hour, col_day = st.columns(2)

    with col_hour:
        st.markdown("#### Violations by Hour of Day")
        if not filtered_df.empty and 'hour' in filtered_df.columns:
            hourly = filtered_df['hour'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(hourly.index, hourly.values, alpha=0.3, color='#FF6B35')
            ax.plot(hourly.index, hourly.values, color='#FF6B35', linewidth=2, marker='o')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Violations')
            ax.set_title('Violation Frequency by Hour')
            ax.set_xticks(range(0, 24))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

    with col_day:
        st.markdown("#### Violations by Day of Week")
        if not filtered_df.empty and 'day_of_week' in filtered_df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']
            daily = filtered_df['day_of_week'].value_counts()
            daily = daily.reindex(day_order, fill_value=0)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(daily)), daily.values, color='#4ECDC4')
            ax.set_xticks(range(len(daily)))
            ax.set_xticklabels([d[:3] for d in daily.index])
            ax.set_ylabel('Violations')
            ax.set_title('Violation Frequency by Day')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # ─── Risk Distribution ────────────────────────────────────────
    st.markdown("### ⚡ Risk & Status Distribution")
    col_risk, col_status = st.columns(2)

    with col_risk:
        if not filtered_df.empty and 'risk_category' in filtered_df.columns:
            risk_counts = filtered_df['risk_category'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            colors_risk = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
            ax.pie(risk_counts.values,
                   labels=risk_counts.index,
                   colors=[colors_risk.get(r, '#999') for r in risk_counts.index],
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12})
            ax.set_title('Risk Category Distribution')
            st.pyplot(fig)
            plt.close(fig)

    with col_status:
        if not filtered_df.empty and 'status' in filtered_df.columns:
            status_counts = filtered_df['status'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            colors_status = {'formal_record': '#3498db', 'pending_review': '#e67e22'}
            ax.pie(status_counts.values,
                   labels=[s.replace('_', ' ').title() for s in status_counts.index],
                   colors=[colors_status.get(s, '#999') for s in status_counts.index],
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12})
            ax.set_title('Record Status Distribution')
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # ─── Recent Evidence Records ──────────────────────────────────
    st.markdown("### 📋 Recent Evidence Records")
    recent = filtered_df.sort_values('timestamp', ascending=False).head(20)
    render_evidence_table(
        recent.to_dict('records'),
        "No evidence records match the selected filters.",
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-header">🚦 Traffic-Eye AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Traffic Violation Detection & Analytics</div>',
                unsafe_allow_html=True)

    # Sidebar — shared elements
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/traffic-light.png", width=80)
        st.title("Navigation")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()

    # Tab-based navigation
    tab_live, tab_analytics = st.tabs(["🎬 Live Detection", "📊 Analytics"])

    with tab_live:
        render_live_detection_tab()

    with tab_analytics:
        render_analytics_tab()

    # ─── Footer ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.9rem;'>"
        "🚦 Traffic-Eye AI — Flipkart Gridlock Hackathon 2.0 | "
        "Theme 3: Automated Traffic Violation Detection | "
        "All data from evidence_store (not hardcoded)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
