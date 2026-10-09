"""Final V3 application. Historical V2 UI is opt-in and retains its implementation."""
from pathlib import Path
import runpy
import streamlit as st
from dotenv import load_dotenv
from dashboard_theme import apply_dark_theme
from live_dashboard import render_live_twin
from live_presentation import sidebar_navigation_html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
st.set_page_config(page_title="Intelligent Hospital Digital Twin", page_icon=":material/local_hospital:", layout="wide", initial_sidebar_state="expanded")
apply_dark_theme()
with st.sidebar:
    with st.container(key="v3_sidebar_rail"):
        with st.container(key="v3_sidebar_navigation"):
            if not st.session_state.get("show_legacy_v2", False):
                st.markdown(sidebar_navigation_html(), unsafe_allow_html=True)
        with st.container(key="v3_legacy_entry"):
            st.caption("Historical")
            with st.expander("Legacy V2 Experiments", expanded=False):
                legacy = st.checkbox("Open Historical V2 Experiments", key="show_legacy_v2",
                    help="Preserved scenario/resource-count optimization and review. Separate from the final fixed-capacity live operating-policy workflow.")
if legacy:
    runpy.run_path(str(PROJECT_ROOT / "src/legacy_dashboard_v2.py"), run_name="__main__")
else:
    from live_presentation import apply_operations_theme
    apply_operations_theme()
    render_live_twin(PROJECT_ROOT / "data/synthetic/synthetic_hospital.csv",
        PROJECT_ROOT / "models/random_forest_pipeline.joblib",
        target_controls=True)
