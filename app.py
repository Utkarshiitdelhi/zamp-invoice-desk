"""Invoice desk: Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Invoice desk · Northwind AP", page_icon="🧾", layout="wide",
                   initial_sidebar_state="expanded")

from ap import db  # noqa: E402
from ap.extract import api_key, configured_model  # noqa: E402
from ap.config import GEMINI_MODEL_CANDIDATES  # noqa: E402
from ui import pages, style  # noqa: E402

nav = st.navigation([
    st.Page(pages.process, title="Process invoice", icon=":material/receipt_long:", url_path="process", default=True),
    st.Page(pages.dashboard, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
    st.Page(pages.reference, title="Reference data", icon=":material/table_view:", url_path="reference"),
    st.Page(pages.how_it_works, title="How it works", icon=":material/account_tree:", url_path="how"),
])
style.inject()
db.init()

with st.sidebar:
    st.markdown("### Invoice desk")
    st.caption("Accounts payable for Northwind Retail Pvt Ltd, Mumbai")
    has_key = bool(api_key())
    st.session_state.setdefault("offline", not has_key)
    offline = st.toggle("Offline reader only", key="offline", disabled=not has_key,
                        help="Skip Gemini and read digital PDFs from their text layer. Scanned PDFs will be held.")
    if has_key and not offline:
        model = configured_model() or GEMINI_MODEL_CANDIDATES[0]
        st.markdown(f'<div class="sidebar-engine"><span class="d" style="background:#1E7A4C"></span>'
                    f'Reader: <b>Gemini</b> ({model})<br/>Results cached per file, so reruns are instant and identical.'
                    f'</div>', unsafe_allow_html=True)
    else:
        why = "no API key configured" if not has_key else "offline mode on"
        st.markdown(f'<div class="sidebar-engine"><span class="d" style="background:#A3650F"></span>'
                    f'Reader: <b>offline text-layer parser</b> ({why}). Digital PDFs work; scanned PDFs are held '
                    f'unless already cached.</div>', unsafe_allow_html=True)
    st.toggle("Demo pacing", key="pacing", value=True,
              help="Adds a short pause between stages so viewers can follow. Timings shown are the real compute time.")
    st.divider()
    if st.button("Reset demo", use_container_width=True,
                 help="Reseed the PO ledger and invoice history and clear the run log, so the demo starts from a known state."):
        db.reset()
        st.session_state.pop("last", None)
        st.toast("Demo reset: ledger and history reseeded, run log cleared.")
    runs_count = st.empty()

nav.run()
_n = len(db.runs())
runs_count.caption(f"{_n} run{'s' if _n != 1 else ''} in the log")
