#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced HR Analytics Streamlit Dashboard
Adds:
- Google Inter font & custom CSS with glass cards
- Color-blind friendly palette (Plotly white template)
- Live dark-mode toggle via CSS variables
- KPI metric cards per tab
- AG-Grid tables (graceful fallback)
- Reset filters button, toast notifications, spinners
- Single-file structure & ast.parse safety check
"""
from __future__ import annotations

import ast
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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, LogisticRegression,
)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import xgboost as xgb

# Optional AG-Grid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID = True
except ModuleNotFoundError:
    AGGRID = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────── Page & Theme ───────────────────────
st.set_page_config(
    page_title="💼 HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORWAY = ["#00429d", "#73a2ff", "#2a9d8f", "#ffd166", "#f8961e", "#d62828"]
PLOTLY_TEMPLATE = "plotly_white"

CSS_BASE = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
html,body,[class*="st-"]{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);}
.metric-card{background:var(--card-bg);padding:1rem 1.2rem;border-radius:12px;
box-shadow:0 2px 6px rgba(0,0,0,.15);margin-bottom:.6rem;}
.stButton>button,.stDownloadButton>button{border:none;border-radius:10px;padding:.45rem 1rem;
box-shadow:0 2px 4px rgba(0,0,0,.15);font-weight:600;}
</style>
"""
CSS_LIGHT = ":root{--bg:#ffffff;--text:#000000;--card-bg:rgba(0,0,0,.04);}"
CSS_DARK  = ":root{--bg:#0e1117;--text:#f5f5f7;--card-bg:rgba(255,255,255,.06);}"

def inject_css(dark: bool) -> None:
    st.markdown((CSS_DARK if dark else CSS_LIGHT) + CSS_BASE, unsafe_allow_html=True)

def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE, colorway=COLORWAY,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ────────────────────── Constants ───────────────────────────
DATA_PATH             = Path(__file__).with_name("HR Analytics.csv")
TARGET                = "Attrition"
DEFAULT_REG_TARGET    = "MonthlyIncome"
RANDOM_STATE          = 42

# ────────────────────── Data Helpers ───────────────────────
@st.cache_data
def load_data(upload: Union[str, Path, None] = None) -> pd.DataFrame:
    df = pd.read_csv(upload) if upload else pd.read_csv(DATA_PATH)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype("category")
    return df

def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = df.select_dtypes(include="number").columns.tolist()
    cat = [c for c in df.columns if c not in num]
    return num, cat

def display_df(df: pd.DataFrame) -> None:
    if AGGRID:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_side_bar()
        AgGrid(df, gridOptions=gb.build(), theme="balham", height=320)
    else:
        st.dataframe(df, use_container_width=True)

def kpi_card(label: str, value: str, delta: str | None = None) -> None:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label, value, delta)
    st.markdown("</div>", unsafe_allow_html=True)

def download_link(obj, filename: str, label: str) -> str:
    txt = obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else obj
    b64 = base64.b64encode(txt.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

# ────────────────────── Sidebar ─────────────────────────────
def sidebar_controls():
    dark = st.sidebar.toggle("🌙 Dark mode", key="dark_mode")
    inject_css(dark)
    if st.sidebar.button("🔄 Reset filters"):
        st.session_state.clear()
        st.experimental_rerun()

# ────────────────────── Filters ─────────────────────────────
def universal_filters(df: pd.DataFrame) -> pd.DataFrame:
    num, cat = get_column_types(df)
    out = df.copy()
    with st.sidebar.expander("Numeric Filters"):
        for col in num:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                st.number_input(col, value=float(lo), disabled=True)
            else:
                rng = st.slider(col, float(lo), float(hi),
                                (float(lo), float(hi)), help=col)
                out = out[(out[col] >= rng[0]) & (out[col] <= rng[1])]
    with st.sidebar.expander("Categorical Filters"):
        for col in cat:
            opts = df[col].dropna().unique().tolist()
            sel  = st.multiselect(col, opts, default=opts, help=col)
            out = out[out[col].isin(sel)]
    return out

# ────────────────────── KPI cache ───────────────────────────
@st.cache_data
def compute_kpis(df: pd.DataFrame) -> Dict[str, str]:
    total = len(df)
    attr  = df[TARGET].value_counts().get("Yes", 0)
    rate  = f"{attr/total*100:.1f}%"
    return {"Employees": f"{total:,}", "Attrition Yes": f"{attr:,}", "Attrition Rate": rate}

# ────────────────────── Modeling helpers ────────────────────
def preprocess(df: pd.DataFrame, target: str):
    num, cat = get_column_types(df.drop(columns=[target]))
    X = df.drop(columns=[target])
    y = df[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return X, y, pre

def train_classifiers(X, y, pre):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "GBRT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    res = {}
    for n, m in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", m)]).fit(X_tr, y_tr)
        p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
        res[n] = {
            "pipe": pipe,
            "train": {
                "accuracy":  accuracy_score (y_tr, p_tr),
                "precision": precision_score(y_tr, p_tr, pos_label="Yes", zero_division=0),
                "recall":    recall_score   (y_tr, p_tr, pos_label="Yes", zero_division=0),
                "f1":        f1_score       (y_tr, p_tr, pos_label="Yes", zero_division=0),
            },
            "test": {
                "accuracy":  accuracy_score (y_te, p_te),
                "precision": precision_score(y_te, p_te, pos_label="Yes", zero_division=0),
                "recall":    recall_score   (y_te, p_te, pos_label="Yes", zero_division=0),
                "f1":        f1_score       (y_te, p_te, pos_label="Yes", zero_division=0),
            },
            "proba": pipe.predict_proba(X_te)[:, 1],
            "y_test": y_te,
        }
    return res

# ────────────────────── Tabs ────────────────────────────────
def tab_visualisation(df: pd.DataFrame):
    for (k, v), c in zip(compute_kpis(df).items(), st.columns(3)):
        with c: kpi_card(k, v)
    num, cat = get_column_types(df)

    with st.expander("Attrition by Department"):
        pct = (pd.crosstab(df["Department"], df[TARGET], normalize="index") * 100).reset_index()
        st.plotly_chart(style_fig(px.bar(pct, x="Department", y="Yes",
                                         labels={"Yes": "Attrition %"})),
                        use_container_width=True)

    with st.expander("Age Distribution"):
        st.plotly_chart(style_fig(
            px.histogram(df, x="Age", color=TARGET, nbins=30, barmode="overlay")),
            use_container_width=True)

    with st.expander("Monthly Income vs Job Level"):
        st.plotly_chart(style_fig(
            px.violin(df, x="JobLevel", y="MonthlyIncome", color=TARGET,
                      box=True, points="outliers")),
            use_container_width=True)

    with st.expander("Correlation Heatmap"):
        st.plotly_chart(style_fig(
            px.imshow(df[num].corr(), text_auto=".2f")),
            use_container_width=True)

def tab_classification(df: pd.DataFrame):
    for (k, v), c in zip(compute_kpis(df).items(), st.columns(3)):
        with c: kpi_card(k, v)

    X, y, pre = preprocess(df, TARGET)
    with st.spinner("Training models…"):
        res = train_classifiers(X, y, pre)

    rows = []
    for n, r in res.items():
        rows.append([
            n, *(round(r["train"][m], 3) for m in ("accuracy", "precision", "recall", "f1")),
               *(round(r["test"][m],  3) for m in ("accuracy", "precision", "recall", "f1")),
        ])
    cols = pd.MultiIndex.from_product([["Train", "Test"],
                                       ["Accuracy", "Precision", "Recall", "F1"]])
    display_df(pd.DataFrame(rows, columns=["Model"] + list(cols)).set_index("Model"))

    mdl = st.selectbox("Model for confusion matrix", list(res))
    cm = confusion_matrix(res[mdl]["y_test"],
                          res[mdl]["pipe"].predict(X.iloc[res[mdl]["y_test"].index]),
                          labels=["No", "Yes"])
    st.plotly_chart(style_fig(
        px.imshow(cm, text_auto=True, x=["No", "Yes"], y=["No", "Yes"],
                  labels={"x": "Predicted", "y": "Actual"})),
        use_container_width=False)

    roc_fig = go.Figure()
    for n, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_test"].map({"No": 0, "Yes": 1}), r["proba"])
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n))
    roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash"))
    roc_fig.update_layout(title="ROC Curves")
    st.plotly_chart(style_fig(roc_fig), use_container_width=True)

    st.subheader("Batch Predict")
    up = st.file_uploader("CSV without Attrition", type="csv", key="batch_up")
    if up:
        new_df = pd.read_csv(up)
        best_pipe = max(res.items(), key=lambda kv: kv[1]["test"]["f1"])[1]["pipe"]
        new_df["PredictedAttrition"] = best_pipe.predict(new_df)
        display_df(new_df.head())
        st.download_button("Download predictions", new_df.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv",
                           on_click=lambda: st.toast("📥 Predictions ready"))

def tab_clustering(df: pd.DataFrame):
    for (k, v), c in zip(compute_kpis(df).items(), st.columns(3)):
        with c: kpi_card(k, v)

    num, _ = get_column_types(df)
    X_scaled = StandardScaler().fit_transform(df[num].dropna())
    inertias = [KMeans(k, n_init="auto", random_state=RANDOM_STATE).fit(X_scaled).inertia_
                for k in range(1, 11)]
    st.plotly_chart(style_fig(
        px.line(x=range(1, 11), y=inertias, markers=True,
                labels={"x": "k", "y": "Inertia"}, title="Elbow Method")),
        use_container_width=True)

    k = st.slider("Select k", 2, 10, 3)
    km = KMeans(k, n_init="auto", random_state=RANDOM_STATE).fit(X_scaled)
    df["cluster"] = km.labels_
    persona = df.groupby("cluster").agg(
        {c: ("mean" if c in num else "first") for c in df})
    display_df(persona)

    st.markdown(download_link(df, "clustered_data.csv", "📥 Download clusters"),
                unsafe_allow_html=True)

def tab_association(df: pd.DataFrame):
    for (k, v), c in zip(compute_kpis(df).items(), st.columns(3)):
        with c: kpi_card(k, v)

    cols = st.multiselect("Select 3 categorical columns",
                          df.columns, default=["JobRole", "MaritalStatus", "OverTime"])
    if len(cols) != 3:
        st.warning("Please select exactly three columns.")
        return
    sup  = st.slider("min_support",    0.01, 0.5, 0.05, 0.01)
    conf = st.slider("min_confidence", 0.01, 1.0, 0.30, 0.01)
    lift = st.slider("min_lift",       0.50, 5.0, 1.00, 0.10)

    hot   = pd.get_dummies(df[cols].astype(str))
    rules = association_rules(apriori(hot, min_support=sup, use_colnames=True),
                              metric="confidence", min_threshold=conf)
    rules = rules[rules["lift"] >= lift]\
            .sort_values("confidence", ascending=False).head(10)
    display_df(rules[["antecedents", "consequents",
                      "support", "confidence", "lift"]])
    st.plotly_chart(style_fig(
        px.bar(rules, x=rules.index.astype(str), y="lift",
               title="Lift of Top-10 Rules")),
        use_container_width=True)

def tab_regression(df: pd.DataFrame):
    for (k, v), c in zip(compute_kpis(df).items(), st.columns(3)):
        with c: kpi_card(k, v)

    target = st.selectbox("Numeric target", df.select_dtypes("number").columns,
                          index=df.columns.get_loc(DEFAULT_REG_TARGET))
    y = df[target]; X = df.drop(columns=[target])
    num, cat = get_column_types(X)
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    }
    scores = []
    for n, m in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", m)]).fit(X, y)
        scores.append((n, round(pipe.score(X, y), 3)))
        if n in ("Linear", "Ridge", "Lasso"):
            coefs = pd.DataFrame({"feature": pipe["prep"].get_feature_names_out(),
                                  "coef": m.coef_})\
                    .sort_values("coef", key=np.abs, ascending=False).head(10)
            with st.expander(f"{n} Coefficients"):
                st.plotly_chart(style_fig(
                    px.bar(coefs, x="coef", y="feature", orientation="h")),
                    use_container_width=True)

    display_df(pd.DataFrame(scores, columns=["Model", "R²"]).set_index("Model"))

    dt = Pipeline([("prep", pre), ("mdl", models["Decision Tree"])]).fit(X, y)
    preds = dt.predict(X)
    st.plotly_chart(style_fig(
        px.scatter(x=preds, y=y - preds,
                   labels={"x": "Predicted", "y": "Residual"},
                   title="Residuals – Decision Tree")),
        use_container_width=True)

def tab_retention(df: pd.DataFrame):
    for (k, v), c in zip(compute_kpis(df).items(), st.columns(3)):
        with c: kpi_card(k, v)

    alg = st.selectbox("Model",
                       ["Logistic Regression", "Random Forest", "XGBoost"])
    horizon = st.slider("Horizon (months)", 6, 24, 12)

    if "YearsAtCompany" not in df:
        st.error("YearsAtCompany column missing.")
        return

    tmp = df.copy()
    tmp["Stay"] = (tmp["YearsAtCompany"] * 12 >= horizon).astype(int)
    X, y = tmp.drop(columns=["Stay"]), tmp["Stay"]
    num, cat = get_column_types(X)
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

    if alg == "Logistic Regression":
        mdl = LogisticRegression(max_iter=1000)
    elif alg == "Random Forest":
        mdl = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE)
    else:
        mdl = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                                learning_rate=0.05, n_estimators=500, max_depth=5)

    pipe = Pipeline([("prep", pre), ("mdl", mdl)]).fit(X, y)
    tmp["RetentionProb"] = pipe.predict_proba(X)[:, 1]
    display_df(tmp[["EmployeeNumber", "RetentionProb"]].head())

    st.subheader("Feature Importance")
    feat_names = pipe["prep"].get_feature_names_out()
    if alg == "Logistic Regression":
        importance = np.abs(mdl.coef_[0])
    elif hasattr(mdl, "feature_importances_"):
        importance = mdl.feature_importances_
    else:
        X_trans = pipe["prep"].transform(X)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        explainer = shap.Explainer(mdl, X_trans)
        shap_vals = explainer(X_trans[:200])
        importance = np.abs(shap_vals.values).mean(axis=0)

    if len(importance) != len(feat_names):
        min_len = min(len(importance), len(feat_names))
        importance = importance[:min_len]
        feat_names = feat_names[:min_len]

    imp_df = pd.DataFrame({"feature": feat_names,
                           "importance": importance})\
             .sort_values("importance", ascending=False).head(15)
    st.plotly_chart(style_fig(
        px.bar(imp_df, x="importance", y="feature", orientation="h")),
        use_container_width=True)

    st.markdown(download_link(tmp[["EmployeeNumber", "RetentionProb"]],
                              "retention_predictions.csv",
                              "📥 Download predictions"),
                unsafe_allow_html=True)

# ────────────────────── Main ────────────────────────────────
def main():
    sidebar_controls()

    upload = st.sidebar.file_uploader("Upload HR Analytics CSV", type="csv")
    if upload:
        st.toast("✅ Upload successful", icon="📂")

    with st.spinner("Loading data…"):
        df = load_data(upload)

    df = universal_filters(df)

    st.sidebar.download_button("Download filtered CSV",
                               df.to_csv(index=False).encode(),
                               "filtered_data.csv", "text/csv",
                               on_click=lambda: st.toast("📥 Download started"))

    tabs = st.tabs(["Visualisation", "Classification", "Clustering",
                    "Association Rules", "Regression", "Retention"])
    with tabs[0]: tab_visualisation(df)
    with tabs[1]: tab_classification(df)
    with tabs[2]: tab_clustering(df)
    with tabs[3]: tab_association(df)
    with tabs[4]: tab_regression(df)
    with tabs[5]: tab_retention(df)

# ────────────────────── Safety check ───────────────────────
ast.parse(Path(__file__).read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
