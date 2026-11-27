"""Main Streamlit entrypoint for SqueezeRadar multipage app."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SqueezeRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_Radar.py", title="Radar", icon="📡", default=True),
    st.Page("pages/2_Deep_Dive.py", title="Deep Dive", icon="🔬"),
    st.Page("pages/3_Time_Machine.py", title="Time Machine", icon="⏪"),
    st.Page("pages/4_Evidence.py", title="Evidence", icon="📊"),
    st.Page("pages/5_AI_Analyst.py", title="AI Analyst", icon="🤖"),
    st.Page("pages/6_About.py", title="About", icon="📘"),
]

pg = st.navigation(pages)
pg.run()
