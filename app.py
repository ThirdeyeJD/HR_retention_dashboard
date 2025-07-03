#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Analytics Streamlit Dashboard
Author  : Jaideep (GMBA) — aided by ChatGPT o3
Version : 2025-07-03  (beautified)
"""

# ───────────────────────── Imports ──────────────────────────
import warnings, base64, ast, inspect, sys
from pathlib import Path
from typing import List, Tuple, Dict, Union

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

# Optional enhancements
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────── Page Config ─────────────────────────
st.set_page_config(page_title="HR Analytics Dashboard 🚀",
                   page_icon="🧠",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ────────────────────── Constants ───────────────────────────
DATA_PATH             = Path(__file__).with_name("HR Analytics.csv")
TARGET_CLASSIFICATION = "Attrition"
DEFAULT_REG_TARGET    = "MonthlyIncome"
RANDOM_STATE          = 42
COLORWAY              = ["#0072B5", "#BC3C29", "#E18727", "#20854E",
                         "#7876B1", "#6F99AD", "#D5A400", "#2EABD8"]

# ────────────────────── Utility CSS ─────────────────────────
def _inject_css(dark: bool = False) -> None:
    """Inject custom CSS for typography, cards, and dark/light themes."""
    bg_primary   = "#0e1117" if dark else "#ffffff"
    bg_secondary = "#161b22" if dark else "#f5f5f5"
    text_color   = "#e6edf3" if dark else "#262730"
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {bg_primary};
        color: {text_color};
    }}
    .card {{
        background: {bg_secondary};
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }}
    .stMetric {{
        background: {bg_secondary};
        border-radius: 12px;
        padding: 0.5rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ────────────────────── Helper Functions ────────────────────
def style_fig(fig: "go.Figure") -> "go.Figure":
    """Apply common styling to Plotly figures."""
    fig.update_layout(template="plotly_white", colorway=COLORWAY,
                      margin=dict(t=40, r=20, b=20, l=20))
    return fig

def show_df(df: pd.DataFrame, key: str = None) -> None:
    """Display DataFrame via AG-Grid if available, else st.dataframe."""
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df.head(1000))
        gb.configure_pagination()
        gb.configure_default_column(groupable=True, filter=True, sortable=True)
        AgGrid(df, gridOptions=gb.build(), theme="alpine", key=key, height=400,
               enable_enterprise_modules=False, fit_columns_on_grid_load=True)
    else:
        st.dataframe(df, use_container_width=True)

def _lint_self() -> None:
    """Run ast.parse on own source to guarantee syntax integrity."""
    try:
        src = inspect.getsource(sys.modules[__name__])
        ast.parse(src)
    except SyntaxError as e:
        st.error(f"Internal linting failed: {e}")

# ────────────────────── Data Loading ────────────────────────
@st.cache_data
def load_data(uploaded: Union[str, Path, None] = None) -> pd.DataFrame:
    """Load bundled sample or user-uploaded CSV."""
    df = pd.read_csv(uploaded) if uploaded else pd.read_csv(DATA_PATH)
    if TARGET_CLASSIFICATION in df.columns:
        df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].astype("category")
    return df

def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric     = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

# ──────────────────── Universal Filters ─────────────────────
def universal_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar widgets that filter the DataFrame."""
    st.sidebar.markdown("### Universal Filters")
    numeric, categorical = get_column_types(df)
    df_filt = df.copy()

    # Reset button
    if st.sidebar.button("🔄 Reset filters", help="Clear all filters and reload"):
        st.session_state.clear()
        st.experimental_rerun()

    # Numeric range sliders (skip constants)
    with st.sidebar.expander("Numeric Ranges", expanded=False):
        for col in numeric:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                st.number_input(col, value=float(lo), disabled=True,
                                help="Column has a single unique value")
                continue
            rng = st.slider(col, float(lo), float(hi),
                            (float(lo), float(hi)),
                            key=f"{col}_rng", help=f"Filter {col} range")
            df_filt = df_filt[(df_filt[col] >= rng[0]) &
                              (df_filt[col] <= rng[1])]

    # Categorical multiselects
    with st.sidebar.expander("Categorical Values", expanded=False):
        for col in categorical:
            opts = df[col].dropna().unique().tolist()
            sel  = st.multiselect(col, opts, default=opts,
                                  key=f"{col}_ms",
                                  help=f"Filter {col} categories")
            df_filt = df_filt[df_filt[col].isin(sel)]

    return df_filt

def download_link(obj, filename: str, label: str) -> str:
    """Return a base-64 download link."""
    text = obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else obj
    b64  = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

# ─────────────────── KPI Helpers ────────────────────────────
def kpi_attrition(df: pd.DataFrame):
    total = len(df)
    attr  = df[TARGET_CLASSIFICATION].value_counts().get("Yes", 0)
    rate  = (attr / total * 100) if total else 0
    col1, col2 = st.columns(2)
    col1.metric("Employees", f"{total:,}")
    col2.metric("Attrition %", f"{rate:.2f}%")

def kpi_best_model(res: Dict[str, Dict]):
    best_name, best_res = max(res.items(), key=lambda kv: kv[1]["test"]["f1"])
    st.metric("🏆 Best Model (F1)", f"{best_name}: {best_res['test']['f1']:.3f}")

def kpi_clusters(k: int):
    st.metric("Clusters", k)

# ──────────────────── Tab 1 – Visualisation ─────────────────
def tab_visualisation(df: pd.DataFrame) -> None:
    kpi_attrition(df)
    st.header("📊 Data Visualisation")
    numeric, categorical = get_column_types(df)

    with st.expander("Attrition % by Department"):
        with st.spinner("Generating…"):
            perc = (pd.crosstab(df["Department"], df[TARGET_CLASSIFICATION],
                                normalize="index")*100).reset_index()
            fig = px.bar(perc, x="Department", y="Yes",
                         labels={"Yes":"Attrition %"})
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander("Age Distribution"):
        with st.spinner("Generating…"):
            fig = px.histogram(df, x="Age", color=TARGET_CLASSIFICATION,
                               nbins=30, barmode="overlay")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander("Monthly Income vs Job Level"):
        with st.spinner("Generating…"):
            fig = px.violin(df, x="JobLevel", y="MonthlyIncome",
                            color=TARGET_CLASSIFICATION,
                            box=True, points="outliers")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander("Correlation Heat-map"):
        with st.spinner("Generating…"):
            fig = px.imshow(df[numeric].corr(), text_auto=".2f")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    auto = 4
    for col in categorical[:8]:
        with st.expander(f"Countplot – {col}"):
            fig = px.histogram(df, x=col, color=TARGET_CLASSIFICATION,
                               barmode="group")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            auto += 1
    for col in numeric[:8]:
        with st.expander(f"Boxplot – {col} by Attrition"):
            fig = px.box(df, y=col, color=TARGET_CLASSIFICATION)
            st.plotly_chart(style_fig(fig), use_container_width=True)
            auto += 1
    st.success(f"Rendered **{auto}** visual insights.")

# ──────────────── Tab 2 – Classification ───────────────────
def preprocess(df: pd.DataFrame, target: str):
    num, cat = get_column_types(df.drop(columns=[target]))
    X = df.drop(columns=[target])
    y = df[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return X, y, pre

@st.cache_data
def train_classifiers_cached(X, y, pre) -> Dict[str, Dict]:
    return train_classifiers(X, y, pre)

def train_classifiers(X, y, pre) -> Dict[str, Dict]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                                                random_state=RANDOM_STATE),
        "GBRT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    out = {}
    for n, m in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", m)]).fit(X_tr, y_tr)
        p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
        out[n] = {
            "pipe": pipe,
            "train": {k: f(y_tr, p_tr, pos_label="Yes", zero_division=0)
                      if k != "accuracy" else f(y_tr, p_tr)
                      for k, f in [("accuracy",accuracy_score),
                                   ("precision",precision_score),
                                   ("recall",recall_score),
                                   ("f1",f1_score)]},
            "test": {k: f(y_te, p_te, pos_label="Yes", zero_division=0)
                     if k != "accuracy" else f(y_te, p_te)
                     for k, f in [("accuracy",accuracy_score),
                                  ("precision",precision_score),
                                  ("recall",recall_score),
                                  ("f1",f1_score)]},
            "proba": pipe.predict_proba(X_te)[:,1],
            "y_test": y_te,
        }
    return out

def tab_classification(df: pd.DataFrame) -> None:
    st.header("🤖 Classification")
    X, y, pre = preprocess(df, TARGET_CLASSIFICATION)
    with st.spinner("Training models…"):
        res = train_classifiers_cached(X, y, pre)

    kpi_best_model(res)

    rows = []
    for n, r in res.items():
        rows.append([n,
            *(round(r["train"][m],3) for m in ("accuracy","precision","recall","f1")),
            *(round(r["test"][m],3)  for m in ("accuracy","precision","recall","f1")),
        ])
    cols = pd.MultiIndex.from_product([["Train","Test"],
                                       ["Accuracy","Precision","Recall","F1"]])
    show_df(pd.DataFrame(rows, columns=["Model"]+list(cols))
                 .set_index("Model"), key="classification_grid")

    mdl = st.selectbox("Confusion-matrix model", list(res),
                       help="Select model to display confusion matrix")
    y_true = res[mdl]["y_test"]
    y_pred = res[mdl]["pipe"].predict(X.iloc[y_true.index])
    cm = confusion_matrix(y_true, y_pred, labels=["No","Yes"])
    fig_cm = px.imshow(cm, text_auto=True,
                       x=["No","Yes"], y=["No","Yes"], title=f"{mdl} – Confusion Matrix")
    st.plotly_chart(style_fig(fig_cm), use_container_width=False)

    roc_fig = go.Figure()
    for n, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_test"].map({"No":0,"Yes":1}), r["proba"])
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n))
    roc_fig.add_shape(type="line", x0=0,x1=1,y0=0,y1=1,
                      line=dict(dash="dash"))
    roc_fig.update_layout(title="ROC Curves",
                          xaxis_title="FPR",
                          yaxis_title="TPR")
    st.plotly_chart(style_fig(roc_fig), use_container_width=True)

    st.subheader("🔮 Batch prediction")
    up = st.file_uploader("CSV without *Attrition*", type="csv",
                          help="Upload data to predict attrition label")
    if up:
        df_new = pd.read_csv(up)
        best   = max(res.items(), key=lambda kv: kv[1]["test"]["f1"])[1]["pipe"]
        df_new["PredictedAttrition"] = best.predict(df_new)
        show_df(df_new.head(), key="predictions_grid")
        st.download_button("Download predictions",
                           df_new.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
        st.toast("Predictions ready for download! 🎉")

# ──────────────── Tab 3 – Clustering ────────────────────────
def tab_clustering(df: pd.DataFrame) -> None:
    st.header("🕵️‍♀️ Clustering")
    numeric,_ = get_column_types(df)
    X_scaled  = StandardScaler().fit_transform(df[numeric].dropna())

    inertias = [KMeans(k, n_init="auto", random_state=RANDOM_STATE)
                .fit(X_scaled).inertia_ for k in range(1,11)]
    fig_elbow = px.line(x=range(1,11), y=inertias, markers=True,
                        labels={"x":"k","y":"Inertia"},
                        title="Elbow Method")
    st.plotly_chart(style_fig(fig_elbow), use_container_width=True)

    k  = st.slider("Choose k", 2,10,3, help="Number of clusters")
    kpi_clusters(k)
    km = KMeans(k, n_init="auto", random_state=RANDOM_STATE).fit(X_scaled)
    df["cluster"] = km.labels_

    persona = df.groupby("cluster").agg(
        {c: ("mean" if c in numeric else "first") for c in df})
    show_df(persona, key="clusters_grid")

    st.markdown(download_link(df, "clustered_data.csv", "📥 Download clusters"),
                unsafe_allow_html=True)
    st.toast("Clustered data generated ✅")

# ──────────────── Tab 4 – Association Rules ────────────────
def tab_association(df: pd.DataFrame) -> None:
    st.header("🔗 Association Rule Mining")
    cols = st.multiselect("Pick 3 categorical columns",
                          df.columns.tolist(),
                          default=["JobRole","MaritalStatus","OverTime"],
                          help="Exactly three columns required for rule mining")
    if len(cols) != 3:
        st.warning("Exactly three columns required.")
        return

    sup  = st.slider("min_support",    0.01,0.5,0.05,0.01)
    conf = st.slider("min_confidence", 0.01,1.0,0.30,0.01)
    lift = st.slider("min_lift",       0.50,5.0,1.00,0.10)

    with st.spinner("Calculating association rules…"):
        hot   = pd.get_dummies(df[cols].astype(str))
        rules = association_rules(apriori(hot, min_support=sup, use_colnames=True),
                                  metric="confidence", min_threshold=conf)
        rules = rules[rules["lift"]>=lift]\
                .sort_values("confidence", ascending=False).head(10)

    show_df(rules[["antecedents","consequents",
                   "support","confidence","lift"]],
            key="assoc_rules_grid")

    fig_lift = px.bar(rules, x=rules.index.astype(str), y="lift",
                      title="Lift of Top-10 Rules")
    st.plotly_chart(style_fig(fig_lift), use_container_width=True)

# ──────────────── Tab 5 – Regression ───────────────────────
def tab_regression(df: pd.DataFrame) -> None:
    st.header("📈 Regression Insights")
    target = st.selectbox("Target variable",
                          df.select_dtypes("number").columns,
                          index=df.columns.get_loc(DEFAULT_REG_TARGET) \
                                if DEFAULT_REG_TARGET in df.columns else 0)
    y = df[target]; X = df.drop(columns=[target])
    num, cat = get_column_types(X)
    pre = ColumnTransformer([("num",StandardScaler(),num),
                             ("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
    models = {
        "Linear"       : LinearRegression(),
        "Ridge"        : Ridge(),
        "Lasso"        : Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=6,
                                               random_state=RANDOM_STATE),
    }
    scores=[]
    for n,m in models.items():
        pipe = Pipeline([("prep",pre),("mdl",m)]).fit(X,y)
        scores.append((n, round(pipe.score(X,y),3)))

        if n in ("Linear","Ridge","Lasso"):
            coefs = pd.DataFrame({"feature":pipe["prep"].get_feature_names_out(),
                                  "coef":m.coef_})\
                    .sort_values("coef", key=np.abs, ascending=False).head(10)
            with st.expander(f"{n} – Top coefficients"):
                fig_coef = px.bar(coefs,x="coef",y="feature",orientation="h",
                                  title=f"{n} – Top coefficients")
                st.plotly_chart(style_fig(fig_coef), use_container_width=True)

    show_df(pd.DataFrame(scores, columns=["Model","R²"]).set_index("Model"),
            key="reg_scores_grid")

    dt = Pipeline([("prep",pre),("mdl",models["Decision Tree"])]).fit(X,y)
    preds = dt.predict(X)
    fig_resid = px.scatter(x=preds,y=y-preds,
                           labels={"x":"Predicted","y":"Residual"},
                           title="Residuals – Decision Tree")
    st.plotly_chart(style_fig(fig_resid), use_container_width=True)

# ──────────────── Tab 6 – Retention ─────────────────────────
def tab_retention(df: pd.DataFrame) -> None:
    st.header("⏳ 12-Month Retention Forecast")
    alg = st.selectbox("Model",
                       ["Logistic Regression","Random Forest","XGBoost"])
    horizon = st.slider("Horizon (months)", 6, 24, 12)

    if "YearsAtCompany" not in df:
        st.error("Column `YearsAtCompany` missing.")
        return

    tmp          = df.copy()
    tmp["Stay"]  = (tmp["YearsAtCompany"]*12 >= horizon).astype(int)
    X,y          = tmp.drop(columns=["Stay"]), tmp["Stay"]
    num,cat      = get_column_types(X)
    pre = ColumnTransformer([("num",StandardScaler(),num),
                             ("cat",OneHotEncoder(handle_unknown="ignore"),cat)])

    if alg == "Logistic Regression":
        mdl = LogisticRegression(max_iter=1000)
    elif alg == "Random Forest":
        mdl = RandomForestClassifier(n_estimators=400,
                                     random_state=RANDOM_STATE)
    else:
        mdl = xgb.XGBClassifier(random_state=RANDOM_STATE,
                                eval_metric="logloss",
                                learning_rate=0.05,
                                n_estimators=500,
                                max_depth=5)

    pipe = Pipeline([("prep",pre),("mdl",mdl)]).fit(X,y)
    tmp["RetentionProb"] = pipe.predict_proba(X)[:,1]
    show_df(tmp[["EmployeeNumber","RetentionProb"]].head(),
            key="retention_grid")

    # ---------- Feature importance ----------
    st.subheader("Feature importance")
    feat_names = pipe["prep"].get_feature_names_out()

    if alg == "Logistic Regression":
        importance = np.abs(mdl.coef_[0])
    elif hasattr(mdl, "feature_importances_"):
        importance = mdl.feature_importances_
    else:  # fallback (rare)
        X_trans = pipe["prep"].transform(X)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        explainer = shap.Explainer(mdl, X_trans)
        shap_vals = explainer(X_trans[:200])
        importance = np.abs(shap_vals.values).mean(axis=0)

    # Align lengths if encoder truncated rare categories
    if len(importance) != len(feat_names):
        min_len = min(len(importance), len(feat_names))
        importance = importance[:min_len]
        feat_names = feat_names[:min_len]

    imp_df = pd.DataFrame({"feature":feat_names,
                           "importance":importance})\
             .sort_values("importance", ascending=False).head(15)

    fig_imp = px.bar(imp_df, x="importance", y="feature",
                     orientation="h", title=f"{alg} – Feature Importance")
    st.plotly_chart(style_fig(fig_imp), use_container_width=True)

    st.markdown(download_link(tmp[[
        "EmployeeNumber","RetentionProb"]],
        "retention_predictions.csv",
        "📥 Download predictions"),
        unsafe_allow_html=True)
    st.toast("Retention probabilities ready ⏳")

# ─────────────────────────── Main ───────────────────────────
def main() -> None:
    # ────────── Dark Mode Toggle & CSS ──────────
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    dark_toggle = st.sidebar.checkbox("🌙 Dark mode", 
                                      value=st.session_state["dark_mode"],
                                      help="Toggle dark / light theme")
    if dark_toggle != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = dark_toggle
        st.experimental_rerun()
    _inject_css(st.session_state["dark_mode"])

    # ────────── Data Source ──────────
    st.sidebar.title("📂 Data Source")
    upload = st.sidebar.file_uploader("Upload HR Analytics CSV", type="csv",
                                      help="Upload your own CSV to override sample")
    df_raw = load_data(upload)
    if upload:
        st.toast("File uploaded successfully! 🚀")
    df     = universal_filters(df_raw)

    st.sidebar.download_button("Download filtered CSV",
                               df.to_csv(index=False).encode(),
                               "filtered_data.csv", "text/csv")
    st.sidebar.caption("Made with ❤️ by Jaideep + ChatGPT o3")

    tabs = st.tabs(["Visualisation","Classification","Clustering",
                    "Association Rules","Regression","12-Month Forecast"])

    with tabs[0]: tab_visualisation(df)
    with tabs[1]: tab_classification(df)
    with tabs[2]: tab_clustering(df)
    with tabs[3]: tab_association(df)
    with tabs[4]: tab_regression(df)
    with tabs[5]: tab_retention(df)

    _lint_self()  # safety

# ──────────────────────────── Run ───────────────────────────
if __name__ == "__main__":
    main()
