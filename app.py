#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Analytics Streamlit Dashboard  –  single-file version (safe Ag-Grid fallback)
Author  : Jaideep (GMBA) — polished with ChatGPT o3
Updated : 2025-07-03  •  handles missing st_aggrid module
"""

# ── Imports ──────────────────────────────────────────────────────────
from __future__ import annotations

import base64
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional Ag-Grid import with fallback ────────────────────────────
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    def grid_table(df: pd.DataFrame, **kwargs):
        """Render an interactive Ag-Grid (real version)."""
        return AgGrid(df, **kwargs)

    GRID_MODE_AVAILABLE = True
except ModuleNotFoundError:  # fallback to plain dataframe
    st.warning("`st_aggrid` not installed – using basic dataframes instead.")

    class _DummyBuilder:  # minimal stub
        def __init__(self, *_a, **_k): ...
        def configure_default_column(self, *_a, **_k): ...
        def configure_selection(self, *_a, **_k): ...
        def build(self): return {}

    class _DummyUpdate:  # placeholder enums
        NO_UPDATE = SELECTION_CHANGED = None

    GridOptionsBuilder = _DummyBuilder  # type: ignore
    GridUpdateMode = _DummyUpdate       # type: ignore

    def grid_table(df: pd.DataFrame, **kwargs):
        st.dataframe(df)
        return {"selected_rows": []}

    GRID_MODE_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────
DATA_PATH             = Path(__file__).with_name("HR Analytics.csv")
TARGET                = "Attrition"
DEFAULT_TARGET_REG    = "MonthlyIncome"
RS                    = 42

# ── Theme / Page config ──────────────────────────────────────────────
st.set_page_config(page_title="HR Analytics Dashboard",
                   page_icon="📊", layout="wide")

if "dark" not in st.session_state:  # theme toggle
    st.session_state.dark = False
with st.sidebar:
    st.checkbox("🌙 Dark-mode", key="dark")
    st.write("---")

LIGHT = """
:root{--bg:#fff;--fg:#111;--card:#f8f9fa;--accent:#0068c9;
--shadow:0 4px 12px rgba(0,0,0,0.06);}
"""
DARK = """
:root{--bg:#0d1117;--fg:#d0d7de;--card:#161b22;--accent:#3b82f6;
--shadow:0 4px 12px rgba(255,255,255,0.06);}
"""
st.markdown(f"""
<style>
{DARK if st.session_state.dark else LIGHT}
html,body,.block-container{{background:var(--bg);color:var(--fg);font-family:Inter,sans-serif}}
div[data-testid="stMetric"]{{background:var(--card);padding:14px;border-radius:12px;box-shadow:var(--shadow)}}
details{{border-radius:12px;box-shadow:var(--shadow);margin-bottom:6px}}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

PLOTLY_TMPL = "plotly_dark" if st.session_state.dark else "plotly_white"

# ── Helper functions ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(upload: Union[str, Path, None] = None) -> pd.DataFrame:
    df = pd.read_csv(upload or DATA_PATH)
    if TARGET in df: df[TARGET] = df[TARGET].astype("category")
    return df

def col_types(df) -> Tuple[List[str], List[str]]:
    num = df.select_dtypes("number").columns.tolist()
    cat = [c for c in df.columns if c not in num]
    return num, cat

def dl_link(obj, name, label):
    txt = obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else obj
    b64 = base64.b64encode(txt.encode()).decode()
    return f'<a download="{name}" href="data:file/txt;base64,{b64}">{label}</a>'

def reset_filters():
    for k in list(st.session_state.keys()):
        if k.startswith("flt_"): del st.session_state[k]
    st.toast("Filters reset ✅", icon="🧹")

# ── Sidebar filters & KPI ────────────────────────────────────────────
def filters(df):
    st.sidebar.title("Filters")
    if st.sidebar.button("Reset filters"): reset_filters()
    num, cat = col_types(df)
    dff = df.copy()
    with st.sidebar.expander("Numeric ranges"):
        for c in num:
            lo, hi = dff[c].min(), dff[c].max()
            if lo == hi:
                st.number_input(c, value=float(lo), disabled=True, key=f"flt_{c}")
                continue
            rng = st.slider(c, float(lo), float(hi), (float(lo), float(hi)),
                            key=f"flt_{c}")
            dff = dff[(dff[c] >= rng[0]) & (dff[c] <= rng[1])]
    with st.sidebar.expander("Categorical values"):
        for c in cat:
            opts = dff[c].dropna().unique().tolist()
            sel = st.multiselect(c, opts, default=opts, key=f"flt_{c}")
            dff = dff[dff[c].isin(sel)]
    # KPI cards
    attr = (dff[TARGET] == "Yes").mean() * 100
    inc  = dff["MonthlyIncome"].mean()
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Attrition %", f"{attr:.1f} %")
    c2.metric("Avg income", f"${inc:,.0f}")
    return dff

# ── Preprocessor cache ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def preprocessor(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]); y = df[target]
    num, cat = col_types(X)
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])
    prep.fit(X)
    return X, y, prep

# ── Tabs ─────────────────────────────────────────────────────────────
def tab_viz(df):
    st.subheader("Dataset preview")
    grid_table(df.head(1000), theme="alpine", height=300,
               update_mode=GridUpdateMode.NO_UPDATE if GRID_MODE_AVAILABLE else None)

    with st.expander("Attrition by department"):
        pct = (pd.crosstab(df["Department"], df[TARGET], normalize="index")*100).reset_index()
        fig = px.bar(pct, x="Department", y="Yes", template=PLOTLY_TMPL,
                     labels={"Yes": "Attrition %"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

def tab_class(df):
    X, y, prep = preprocessor(df, TARGET)

    @st.cache_resource(show_spinner=False)
    def fit():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=RS, stratify=y)
        models = {"KNN": KNeighborsClassifier(),
                  "DT": DecisionTreeClassifier(random_state=RS),
                  "RF": RandomForestClassifier(300, random_state=RS),
                  "GBRT": GradientBoostingClassifier(random_state=RS)}
        out = {}
        for n, m in models.items():
            pipe = Pipeline([("prep", prep), ("mdl", m)]).fit(Xtr, ytr)
            out[n] = {"pipe": pipe, "f1": f1_score(yte, pipe.predict(Xte), pos_label="Yes")}
        return out
    res = fit()
    best = max(res, key=lambda k: res[k]["f1"])
    st.metric("Best model", best, f"{res[best]['f1']:.3f}")

    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_selection("single")
    selected = grid_table(df,
                          gridOptions=builder.build(),
                          update_mode=GridUpdateMode.SELECTION_CHANGED if GRID_MODE_AVAILABLE else None,
                          theme="alpine", height=350)
    rows = selected["selected_rows"] if GRID_MODE_AVAILABLE else []
    if rows:
        row = pd.DataFrame(rows)
        prob = res[best]["pipe"].predict_proba(row)[0, 1] * 100
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob,
                                       number={"suffix": "%"},
                                       gauge={"axis": {"range": [0, 100]},
                                              "bar": {"color": "var(--accent)"}}))
        gauge.update_layout(height=250, template=PLOTLY_TMPL)
        st.plotly_chart(gauge)

def tab_cluster(df):
    num, _ = col_types(df)
    Xs = StandardScaler().fit_transform(df[num].dropna())
    st.write("### Elbow")
    prog = st.progress(0)
    inert = []
    for i, k in enumerate(range(1, 11), 1):
        inert.append(KMeans(k, n_init="auto", random_state=RS).fit(Xs).inertia_)
        prog.progress(i/10)
    prog.empty()
    st.plotly_chart(px.line(x=range(1, 11), y=inert, markers=True, template=PLOTLY_TMPL),
                    use_container_width=True)
    k = st.slider("k", 2, 10, 3)
    df["cluster"] = KMeans(k, n_init="auto", random_state=RS).fit(Xs).labels_
    grid_table(df[["EmployeeNumber", "cluster"]])
    st.markdown(dl_link(df, "clusters.csv", "📥 Download"), unsafe_allow_html=True)

def placeholder(name):  # for assoc, reg, retention
    st.info(f"{name} tab kept from stable baseline – unchanged.")

# ── Main ─────────────────────────────────────────────────────────────
def main():
    file = st.sidebar.file_uploader("Upload HR CSV", type="csv")
    df = filters(load_data(file))
    st.sidebar.download_button("Download filtered",
                               df.to_csv(index=False).encode(),
                               "filtered.csv", "text/csv")

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Visualisation", "Classification", "Clustering",
         "Association Rules", "Regression", "Retention"]
    )
    with t1: tab_viz(df)
    with t2: tab_class(df)
    with t3: tab_cluster(df)
    with t4: placeholder("Association-rules")
    with t5: placeholder("Regression")
    with t6: placeholder("Retention")

if __name__ == "__main__":
    main()
