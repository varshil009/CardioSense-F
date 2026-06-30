import io
import json
from pathlib import Path
from typing import Any
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("EC2_ENDPOINT")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SUPERCLASS_MAP = {
    "NORM": "Normal Heart Rhythm",
    "HYP": "Thickened Heart Muscle (Hypertrophy)",
    "STTC": "Changes in Heart Electrical Signals (ST-T Changes)",
    "CD": "Heart's Electrical Conduction Problem",
    "MI": "Heart Attack (Myocardial Infarction)",
}

st.set_page_config(page_title="ECG Diagnosis Application", layout="wide", initial_sidebar_state="collapsed")

# Compact CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2.5rem !important; padding-bottom: 0.5rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.25rem 1rem; font-size: 0.85rem; }
    h1 { font-size: 1.5rem !important; margin-bottom: 0.2rem !important; }
    h2, h3 { margin-bottom: 0.3rem !important; }
    .st-emotion-cache-1r6slb0 { font-size: 0.85rem; }
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        transition: all 0.2s ease;
    }
    div[data-testid="stDownloadButton"] button:hover {
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.6);
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_test_signals() -> list[dict[str, Any]]:
    """Fetch list of available test signals from the backend."""
    try:
        resp = requests.get(f"{API_BASE}/api/test-signals", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("signals", [])
    except requests.RequestException:
        st.error("Could not connect to backend. Please ensure the backend server is running.")
        return []


def health_check() -> dict[str, Any] | None:  # noqa: UP007
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def download_signal_for_plot(signal_name: str) -> np.ndarray | None:  # noqa: UP007
    """Download a test signal's .npy file from the backend API."""
    try:
        resp = requests.get(f"{API_BASE}/api/test-signal/{signal_name}", timeout=30)
        resp.raise_for_status()
        return np.load(io.BytesIO(resp.content))
    except requests.RequestException:
        signal_path = Path(__file__).resolve().parents[1] / "test_signals" / signal_name
        if signal_path.exists():
            return np.load(signal_path)
        return None


# ──────────────────────────────────────────────────────────────────────
# UI — Flow: Signal Upload → Patient Info → Analysis → Report
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("ECG Diagnosis Application")

    # Backend health check
    health = health_check()
    if health is None:
        st.warning("Backend not reachable. Start: `uvicorn backend.main:app --reload --port 8000`")
    else:
        st.success("Backend connected")
        st.caption("12-lead ECG analysis with SupCon, CrossLeadTransformer, K-RAG, and Groq-assisted reporting.")

    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("signal_data", None)
    st.session_state.setdefault("signal_name", None)
    st.session_state.setdefault("signal_metadata", None)
    st.session_state.setdefault("patient_info", {"age": 45, "sex": "M", "additional_context": ""})
    st.session_state.setdefault("analysis_in_progress", False)

    # ── Main single-page flow ──
    render_signal_section()

    if st.session_state.signal_data is not None and not st.session_state.analysis_in_progress:
        render_patient_info_and_analysis()

    if st.session_state.analysis_in_progress:
        render_analysis_progress()

    if st.session_state.analysis_result is not None and not st.session_state.analysis_in_progress:
        render_results_section()

    # ── Separate tab for ECG visualisation ──
    if st.session_state.signal_data is not None:
        viz_tab = st.tabs(["📊 12-Lead ECG Visualization"])
        with viz_tab[0]:
            render_signal_visualization()


# ──────────────────────────────────────────────────────────────────────
# Step 1: Load a signal
# ──────────────────────────────────────────────────────────────────────

def render_signal_section() -> None:
    st.subheader("1. Load ECG Signal")
    source = st.radio("Signal source", ["Pre-loaded test signal", "Upload file"], horizontal=True)

    if source == "Pre-loaded test signal":
        signals = fetch_test_signals()
        if not signals:
            st.info("No test signals found on backend.")
            return

        if isinstance(signals[0], dict):
            signal_names = [s.get("signal_file", str(s.get("ecg_id", ""))) for s in signals]
        else:
            signal_names = signals

        col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
        with col1:
            signal_name = st.selectbox("Select signal", signal_names, label_visibility="collapsed")
        with col2:
            load_clicked = st.button("Load", use_container_width=True, type="primary")

        if load_clicked:
            signal_data = download_signal_for_plot(signal_name)
            if signal_data is not None:
                if signal_data.ndim != 2 or signal_data.shape[0] != 1000 or signal_data.shape[1] != 12:
                    st.error(
                        f"Invalid shape: {signal_data.shape}. "
                        "The signal must have exactly **1000 time samples × 12 leads** (shape 1000, 12)."
                    )
                    return

                st.session_state.signal_data = signal_data
                st.session_state.signal_name = signal_name

                meta = None
                if isinstance(signals[0], dict):
                    for s in signals:
                        if s.get("signal_file") == signal_name:
                            meta = s
                            break
                st.session_state.signal_metadata = meta
                st.rerun()
            else:
                st.error("Could not load signal file.")

    else:
        st.warning("⚠️ File must be a **.npy** array with shape **(1000, 12)** (1000 time samples × 12 ECG leads).")
        uploaded = st.file_uploader("Upload ECG file", type=["npy", "csv", "mat"], label_visibility="collapsed")
        if uploaded is not None:
            content = uploaded.getvalue()
            try:
                signal_data = np.load(io.BytesIO(content))
            except Exception:
                try:
                    signal_data = pd.read_csv(io.BytesIO(content)).to_numpy()
                except Exception:
                    st.error("Unsupported file format. Please upload .npy, .csv, or .mat files.")
                    return

            if signal_data.ndim != 2 or signal_data.shape[0] != 1000 or signal_data.shape[1] != 12:
                st.error(
                    f"Invalid shape: {signal_data.shape}. "
                    "The signal must have exactly **1000 time samples × 12 leads** (shape 1000, 12)."
                )
                return

            st.session_state.signal_data = signal_data
            st.session_state.signal_name = uploaded.name
            st.session_state.signal_metadata = None
            st.rerun()

    # Show signal information after loading
    if st.session_state.signal_data is not None:
        render_signal_information()


# ──────────────────────────────────────────────────────────────────────
# Display signal metadata (without internal details like filename/shape)
# ──────────────────────────────────────────────────────────────────────

def friendly_name(code: str) -> str:
    """Map a disease code to a user-friendly name, or return the code as-is if unknown."""
    return SUPERCLASS_MAP.get(code.upper(), code)


def render_signal_information() -> None:
    st.markdown("**Actual Signal Information**")
    meta = st.session_state.signal_metadata
    if meta:
        super_code = meta.get("super_class", meta.get("true_superclass", "?"))
        sub_code = meta.get("sub_class", meta.get("true_subclass", "?"))
        st.metric("Heart Condition", friendly_name(super_code))
        cols = st.columns(3)
        cols[0].metric("Subtype", friendly_name(sub_code))
        sex_val = meta.get("sex", meta.get("patient_sex", ""))
        age_val = meta.get("age", meta.get("patient_age", ""))
        cols[1].metric("Sex", str(sex_val) if sex_val else "—")
        cols[2].metric("Age", str(age_val) if age_val else "—")
    else:
        st.info("Signal loaded (no additional metadata available).")


# ──────────────────────────────────────────────────────────────────────
# Step 2: Patient info + trigger analysis
# ──────────────────────────────────────────────────────────────────────

def render_patient_info_and_analysis() -> None:
    st.divider()
    st.subheader("2. Patient Information")

    info = st.session_state.patient_info
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age", min_value=0, max_value=120, value=int(info["age"]))
    sex = col2.selectbox("Sex", ["M", "F"], index=0 if info["sex"] == "M" else 1)
    additional_context = col3.text_input(
        "Clinical context",
        value=info["additional_context"],
        placeholder="Optional notes...",
    )
    st.session_state.patient_info = {
        "age": int(age),
        "sex": sex,
        "additional_context": additional_context.strip(),
    }

    st.divider()
    if st.button("Start Analysis", type="primary", use_container_width=True):
        st.session_state.analysis_in_progress = True
        st.session_state.analysis_result = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
# Step 3: Show analysis progress (replaces patient info section)
# ──────────────────────────────────────────────────────────────────────

def render_analysis_progress() -> None:
    st.subheader("Running Analysis")
    signal_data = st.session_state.signal_data
    signal_name = st.session_state.signal_name
    patient_info = st.session_state.patient_info

    progress_bar = st.progress(0)
    status_box = st.empty()

    buf = io.BytesIO()
    np.save(buf, signal_data)
    buf.seek(0)
    files = {"signal_file": (signal_name or "signal.npy", buf, "application/octet-stream")}

    data = {
        "age": patient_info["age"],
        "sex": patient_info["sex"],
        "additional_context": patient_info["additional_context"],
    }

    try:
        progress_bar.progress(20)
        status_box.write("Sending to backend...")

        resp = requests.post(f"{API_BASE}/api/analyze", files=files, data=data, timeout=300)

        progress_bar.progress(80)
        status_box.write("Processing response...")

        resp.raise_for_status()
        result = resp.json()

        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
            st.session_state.analysis_in_progress = False
            return

        st.session_state.analysis_result = result
        progress_bar.progress(100)
        status_box.write("Analysis completed!")
        st.success("Analysis completed successfully")
        st.session_state.analysis_in_progress = False
        st.rerun()

    except requests.Timeout:
        st.error("Request timed out.")
        st.session_state.analysis_in_progress = False
    except requests.RequestException:
        st.error("Backend request failed. Please try again later.")
        st.session_state.analysis_in_progress = False
    except Exception:
        st.error("An unexpected error occurred. Please try again.")
        st.session_state.analysis_in_progress = False


# ──────────────────────────────────────────────────────────────────────
# Step 4: Show diagnostic report
# ──────────────────────────────────────────────────────────────────────

def render_results_section() -> None:
    st.divider()
    result = st.session_state.analysis_result

    report = result.get("report_payload", {})
    confidences = result.get("superclass_confidences", {})
    sample_name = result.get("sample_name", "report")

    # Header row with title and PDF button
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.subheader("3. Diagnostic Report")
    with hdr_col2:
        st.write("")
        try:
            pdf_resp = requests.post(
                f"{API_BASE}/api/render-report",
                json={"report_payload": report},
                timeout=30,
            )
            if pdf_resp.status_code == 200:
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_resp.content,
                    file_name=f"{sample_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.error("PDF unavailable")
        except Exception:
            st.error("PDF unavailable")

    # Top summary rows
    superclass = result.get("superclass", "?")
    top_conf = max(confidences.values()) if confidences else 0
    st.metric("Predicted Condition", friendly_name(superclass), f"{top_conf * 100:.2f}%")

    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    col1.metric("Primary Implication", result.get("primary_implication", "N/A")[:40])
    patient_info_report = report.get("patient_information", {})
    col2.metric("Age", patient_info_report.get("age", "?"))
    col3.metric("Sex", patient_info_report.get("sex", "?"))

    # Superclass confidence table
    if confidences:
        conf_df = pd.DataFrame(
            {"Condition": [friendly_name(k) for k in confidences.keys()],
             "Code": list(confidences.keys()),
             "Confidence %": [f"{v * 100:.2f}%" for v in confidences.values()]}
        )
        st.dataframe(conf_df, hide_index=True, use_container_width=True)

    # Implications and findings
    imp_col, diag_col = st.columns(2)
    with imp_col:
        st.write("**Possible implications**")
        for imp in result.get("possible_implications", [])[:5]:
            st.write(f"- {imp.get('label', '?')}")
    with diag_col:
        st.write("**Key findings**")
        for finding in result.get("subclass_reasoning", [])[:5]:
            st.write(f"- {finding}")

    st.write("**Knowledge-based analysis**")
    llm_analysis = result.get("llm_analysis", {})
    diagnosis_text = report.get("knowledge_based_analysis", llm_analysis).get("primary_diagnosis", "")
    with st.expander("View full analysis", expanded=len(diagnosis_text) <= 500):
        st.write(diagnosis_text)
    limitations = llm_analysis.get("limitations", "")
    if limitations:
        st.caption(limitations)

    abbreviations = report.get("abbreviation_footnotes", [])
    if abbreviations:
        with st.expander("Abbreviation Footnotes", expanded=False):
            for abbr, full_form in abbreviations[:20]:
                st.write(f"- {abbr} = {full_form}")

    # Allow starting a new analysis
    st.divider()
    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.signal_data = None
        st.session_state.signal_name = None
        st.session_state.signal_metadata = None
        st.session_state.analysis_in_progress = False
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
# ECG visualisation (in a dedicated tab)
# ──────────────────────────────────────────────────────────────────────

def render_signal_visualization() -> None:
    signal_data = st.session_state.signal_data
    if signal_data is None:
        st.info("Load a signal first.")
        return

    if signal_data.ndim != 2 or signal_data.shape[1] < 12:
        st.error(f"Invalid shape: {signal_data.shape}. Expected (samples, 12).")
        return

    fig, axes = plt.subplots(6, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        axes[lead_idx].plot(signal_data[:, lead_idx], linewidth=0.6, color="#1f77b4")
        axes[lead_idx].set_title(lead_name, fontsize=9)
        axes[lead_idx].grid(alpha=0.15)
        axes[lead_idx].tick_params(labelsize=7)
    fig.tight_layout(pad=0.5, h_pad=0.4, w_pad=0.3)
    st.pyplot(fig)


if __name__ == "__main__":
    main()