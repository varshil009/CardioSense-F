import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

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
    except requests.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return []


def health_check() -> dict[str, Any] | None:  # noqa: UP007
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# ──────────────────────────────────────────────────────────────────────
# Signal download helper (for plotting)
# ──────────────────────────────────────────────────────────────────────

def download_signal_for_plot(signal_name: str) -> np.ndarray | None:  # noqa: UP007
    """Download a test signal's .npy file from the local filesystem."""
    signal_path = Path(__file__).resolve().parents[1] / "test_signals" / signal_name
    if signal_path.exists():
        return np.load(signal_path)
    return None


# ──────────────────────────────────────────────────────────────────────
# UI
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

    tabs = st.tabs(
        [
            "1. Signal Upload",
            "2. Patient Info",
            "3. Visualization",
            "4. Analysis",
            "5. Results",
        ]
    )

    with tabs[0]:
        render_signal_upload()
    with tabs[1]:
        render_patient_information()
    with tabs[2]:
        render_signal_visualization()
    with tabs[3]:
        render_analysis_dashboard()
    with tabs[4]:
        render_results()


def render_signal_upload() -> None:
    st.subheader("Load ECG signal")
    source = st.radio("Signal source", ["Pre-loaded test signal", "Upload file"], horizontal=True)

    if source == "Pre-loaded test signal":
        signals = fetch_test_signals()
        if not signals:
            st.info("No test signals found on backend.")
            return

        # Handle both dict list and string list formats
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
                st.session_state.signal_data = signal_data
                st.session_state.signal_name = signal_name

                # Store metadata if available
                meta = None
                if isinstance(signals[0], dict):
                    for s in signals:
                        if s.get("signal_file") == signal_name:
                            meta = s
                            break
                st.session_state.signal_metadata = meta
                st.success(f"Loaded {signal_name} ({signal_data.shape})")
            else:
                st.error(f"Could not load signal file: {signal_name}")

    else:
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

            st.session_state.signal_data = signal_data
            st.session_state.signal_name = uploaded.name
            st.session_state.signal_metadata = None
            st.success(f"Loaded {uploaded.name} ({signal_data.shape})")

    # Display loaded signal info + metadata
    signal_data = st.session_state.signal_data
    if signal_data is not None:
        meta = st.session_state.signal_metadata
        info_cols = st.columns(4)
        info_cols[0].metric("File", st.session_state.signal_name[:20])
        info_cols[1].metric("Shape", f"{signal_data.shape[0]}×{signal_data.shape[1]}")
        if meta:
            info_cols[2].metric("True Superclass", meta.get("super_class", meta.get("true_superclass", "?")))
            info_cols[3].metric("True Subclass", meta.get("sub_class", meta.get("true_subclass", "?")))
            sex_val = meta.get("sex", meta.get("patient_sex", ""))
            age_val = meta.get("age", meta.get("patient_age", ""))
            info_cols = st.columns(4)
            info_cols[0].metric("Sex", str(sex_val) if sex_val else "—")
            info_cols[1].metric("Age", str(age_val) if age_val else "—")
        else:
            info_cols[2].metric("Dtype", str(signal_data.dtype))


def render_patient_information() -> None:
    st.subheader("Patient information")
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


def render_signal_visualization() -> None:
    st.subheader("12-lead ECG")
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


def render_analysis_dashboard() -> None:
    st.subheader("Run analysis")
    signal_data = st.session_state.signal_data
    signal_name = st.session_state.signal_name

    if signal_data is None:
        st.info("Load a signal first.")
        return

    patient_info = st.session_state.patient_info

    if st.button("Start analysis", type="primary", use_container_width=True):
        st.info("Sending signal to backend...")
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
                return

            st.session_state.analysis_result = result
            progress_bar.progress(100)
            status_box.write("Analysis completed!")
            st.success("Analysis completed successfully")

        except requests.Timeout:
            st.error("Request timed out.")
        except requests.RequestException as exc:
            st.error(f"Backend request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def render_results() -> None:
    st.subheader("Diagnostic report")
    result = st.session_state.analysis_result
    if result is None:
        st.info("Run the analysis first.")
        return

    report = result.get("report_payload", {})
    confidences = result.get("superclass_confidences", {})
    sample_name = result.get("sample_name", "report")

    # Top summary row with PDF button near Sex
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 1.5, 2])
    superclass = result.get("superclass", "?")
    top_conf = max(confidences.values()) if confidences else 0
    col1.metric("Superclass", superclass, f"{top_conf * 100:.2f}%")
    col2.metric("Top Implication", result.get("primary_implication", "N/A")[:25])
    patient_info_report = report.get("patient_information", {})
    col3.metric("Age", patient_info_report.get("age", "?"))
    col4.metric("Sex", patient_info_report.get("sex", "?"))
    with col5:
        st.write("")
        st.write("")
        try:
            pdf_resp = requests.post(
                f"{API_BASE}/api/render-report",
                json={"report_payload": report},
                timeout=30,
            )
            if pdf_resp.status_code == 200:
                st.download_button(
                    "Download PDF",
                    data=pdf_resp.content,
                    file_name=f"{sample_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.error("PDF unavailable")
        except Exception:
            st.error("PDF unavailable")

    # Superclass confidence table (compact)
    if confidences:
        conf_df = pd.DataFrame(
            {"Superclass": list(confidences.keys()), "Confidence %": [f"{v * 100:.2f}%" for v in confidences.values()]}
        )
        st.dataframe(conf_df, hide_index=True, use_container_width=True)

    # Implications, findings, diagnosis
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


if __name__ == "__main__":
    main()