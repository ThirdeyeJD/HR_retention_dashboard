# app.py
import streamlit as st
import pandas as pd
from utils.data_loader import load_data

# ───────────────────────────────────────────────
# Page configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="HR Analytics Suite",
    page_icon="📊",
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/<you>/hr-analytics-dashboard/issues",
        "About": "A full-stack Streamlit Cloud solution for HR attrition analytics, clustering, association rules, regression insights, and 12-month retention forecasting.",
    },
)

# ───────────────────────────────────────────────
# Sidebar: dataset loader, global filters, download-all
# ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # 1 — upload alternative CSV (optional)
    uploaded_file = st.file_uploader(
        "Replace default **HR Analytics.csv** (optional)",
        type=["csv"],
        help="Must keep the exact column headers specified in the README.",
    )
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state["source"] = "uploaded CSV"
    else:
        raw_df = load_data()  # default file from repo root
        st.session_state["source"] = "HR Analytics.csv"

    # 2 — universal filters (extend at will)
    st.subheader("🔎 Universal Filters")

    gender_opts = raw_df["Gender"].unique().tolist()
    gender_sel = st.multiselect(
        "Gender",
        options=gender_opts,
        default=gender_opts,
        help="Filter all pages to these gender(s).",
    )

    dept_opts = raw_df["Department"].unique().tolist()
    dept_sel = st.multiselect(
        "Department",
        options=dept_opts,
        default=dept_opts,
        help="Filter all pages to these department(s).",
    )

    mask = raw_df["Gender"].isin(gender_sel) & raw_df["Department"].isin(dept_sel)
    filtered_df = raw_df[mask]

    # persist for other pages
    st.session_state["filtered_df"] = filtered_df

    # 3 — download-all button
    st.download_button(
        "⬇️ Download filtered data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_HR_Analytics.csv",
        help="CSV reflects all active sidebar filters.",
    )

    st.markdown("---")
    st.info(
        f"Data source: **{st.session_state['source']}**  \n"
        f"Rows after filters: **{len(filtered_df):,}**",
        icon="📄",
    )

# ───────────────────────────────────────────────
# Main landing page
# ───────────────────────────────────────────────
st.title("🏢 HR Analytics & Retention Dashboard")

st.write(
    """
Welcome to the **HR Analytics Suite**.  
Navigate with the sidebar or the **Pages** menu to explore:

1. **📊 Data Visualisation** – 20 + rich descriptive insights  
2. **🤖 Classification** – four-model attrition predictor with ROC & batch scoring  
3. **🔍 Clustering** – K-means personas & elbow chart  
4. **🔗 Association Rules** – Apriori mining on any three categorical columns  
5. **📈 Regression Insights** – pay & performance drivers via multiple regressors  
6. **⏳ Retention Forecast** – probability each employee stays ≥ 12 months
"""
)

with st.expander("🗄️ Preview first 200 rows (after filters)"):
    st.dataframe(filtered_df.head(200), use_container_width=True)

st.success("Dataset loaded and global filters applied ✅")
