#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Analytics Streamlit Dashboard
Author  : Group 8 — aided by ChatGPT o3
Updated : 2025-07-03
• Handles constant-value numeric columns in sliders
• Fixes metrics-table column mismatch
• Restores correct tab routing
"""

import warnings, base64
from pathlib import Path
from typing import List, Tuple, Dict

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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="HR Analytics Dashboard",
                   page_icon=":bar_chart:", layout="wide")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).with_name("HR Analytics.csv")
TARGET_CLASSIFICATION = "Attrition"
DEFAULT_REG_TARGET = "MonthlyIncome"
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)
    if TARGET_CLASSIFICATION in df.columns:
        df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].astype("category")
    return df


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def universal_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar widgets that filter the DataFrame."""
    st.sidebar.markdown("### Universal Filters")
    numeric, categorical = get_column_types(df)
    df_filt = df.copy()

    # Numeric sliders (guard against constant columns)
    with st.sidebar.expander("Numeric Ranges"):
        for col in numeric:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                st.number_input(
                    col,
                    value=float(lo),
                    disabled=True,
                    help=f"{col} is constant ({lo})",
                )
                continue
            sel_lo, sel_hi = st.slider(
                col,
                float(lo),
                float(hi),
                (float(lo), float(hi)),
                help=f"Filter {col}",
            )
            df_filt = df_filt[(df_filt[col] >= sel_lo) & (df_filt[col] <= sel_hi)]

    # Categorical multiselects
    with st.sidebar.expander("Categorical Values"):
        for col in categorical:
            opts = df[col].dropna().unique().tolist()
            sel = st.multiselect(col, opts, default=opts)
            df_filt = df_filt[df_filt[col].isin(sel)]

    return df_filt


def download_link(obj, filename, label):
    txt = obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else obj
    b64 = base64.b64encode(txt.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

# ──────────────────────────────────────────────────────────────────────
# Tab 1 – Data Visualisation
# ──────────────────────────────────────────────────────────────────────
def tab_visualisation(df: pd.DataFrame):
    st.header("📊 Data Visualisation")
    numeric, categorical = get_column_types(df)

    with st.expander("Attrition % by Department"):
        ctab = (pd.crosstab(df["Department"], df[TARGET_CLASSIFICATION],
                            normalize="index") * 100).reset_index()
        st.plotly_chart(px.bar(ctab, x="Department", y="Yes",
                               labels={"Yes": "Attrition %"}),
                        use_container_width=True)

    with st.expander("Age Distribution"):
        st.plotly_chart(
            px.histogram(df, x="Age", color=TARGET_CLASSIFICATION,
                         nbins=30, barmode="overlay"),
            use_container_width=True,
        )

    with st.expander("Monthly Income vs Job Level"):
        st.plotly_chart(
            px.violin(df, x="JobLevel", y="MonthlyIncome",
                      color=TARGET_CLASSIFICATION,
                      box=True, points="outliers"),
            use_container_width=True,
        )

    with st.expander("Correlation Heat-map"):
        st.plotly_chart(px.imshow(df[numeric].corr(), text_auto=".2f"),
                        use_container_width=True)

    auto = 4
    for col in categorical[:8]:
        with st.expander(f"Countplot – {col}"):
            st.plotly_chart(
                px.histogram(df, x=col, color=TARGET_CLASSIFICATION,
                             barmode="group"),
                use_container_width=True,
            )
            auto += 1

    for col in numeric[:8]:
        with st.expander(f"Boxplot – {col} by Attrition"):
            st.plotly_chart(
                px.box(df, y=col, color=TARGET_CLASSIFICATION),
                use_container_width=True,
            )
            auto += 1

    st.success(f"Rendered **{auto}** visual insights.")

# ──────────────────────────────────────────────────────────────────────
# Tab 2 – Classification
# ──────────────────────────────────────────────────────────────────────
def preprocess_for_modeling(df: pd.DataFrame, target: str):
    numeric, categorical = get_column_types(df.drop(columns=[target]))
    X = df.drop(columns=[target])
    y = df[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return X, y, pre


def train_classifiers(X, y, pre) -> Dict[str, Dict]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE),
        "GBRT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    res = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", model)]).fit(X_tr, y_tr)
        res[name] = {
            "pipe": pipe,
            "train": {
                "accuracy": accuracy_score(y_tr, pipe.predict(X_tr)),
                "precision": precision_score(y_tr, pipe.predict(X_tr), pos_label="Yes"),
                "recall": recall_score(y_tr, pipe.predict(X_tr), pos_label="Yes"),
                "f1": f1_score(y_tr, pipe.predict(X_tr), pos_label="Yes"),
            },
            "test": {
                "accuracy": accuracy_score(y_te, pipe.predict(X_te)),
                "precision": precision_score(y_te, pipe.predict(X_te), pos_label="Yes"),
                "recall": recall_score(y_te, pipe.predict(X_te), pos_label="Yes"),
                "f1": f1_score(y_te, pipe.predict(X_te), pos_label="Yes"),
            },
            "proba": pipe.predict_proba(X_te)[:, 1],
            "y_test": y_te,
        }
    return res


def tab_classification(df: pd.DataFrame):
    st.header("🤖 Classification")
    X, y, pre = preprocess_for_modeling(df, TARGET_CLASSIFICATION)
    res = train_classifiers(X, y, pre)

    # Metrics table (fixed columns)
    rows = []
    for name, r in res.items():
        rows.append([
            name,
            *(round(r["train"][m], 3) for m in ("accuracy", "precision", "recall", "f1")),
            *(round(r["test"][m], 3)  for m in ("accuracy", "precision", "recall", "f1")),
        ])
    cols_nested = pd.MultiIndex.from_product(
        [["Train", "Test"], ["Accuracy", "Precision", "Recall", "F1"]]
    )
    metrics_df = pd.DataFrame(rows, columns=["Model"] + list(cols_nested))\
                 .set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)

    # Confusion matrix
    model_sel = st.selectbox("Confusion-matrix model", list(res.keys()))
    y_true = res[model_sel]["y_test"]
    y_pred = res[model_sel]["pipe"].predict(X.iloc[y_true.index])
    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])
    st.plotly_chart(
        px.imshow(cm, text_auto=True, x=["No", "Yes"], y=["No", "Yes"],
                  labels=dict(x="Predicted", y="Actual")),
        use_container_width=False,
    )

    # ROC curves
    fig_roc = go.Figure()
    for name, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_test"].map({"No": 0, "Yes": 1}), r["proba"])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash"))
    fig_roc.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Batch prediction
    st.subheader("🔮 Batch prediction")
    up = st.file_uploader("CSV without *Attrition*", type="csv")
    if up:
        new_df = pd.read_csv(up)
        best = max(res.items(), key=lambda kv: kv[1]["test"]["f1"])[1]["pipe"]
        new_df["PredictedAttrition"] = best.predict(new_df)
        st.dataframe(new_df.head(), use_container_width=True)
        st.download_button("Download predictions",
                           new_df.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────
# Tab 3 – Clustering
# ──────────────────────────────────────────────────────────────────────
def tab_clustering(df: pd.DataFrame):
    st.header("🕵️‍♀️ Clustering")
    numeric, _ = get_column_types(df)
    X_scaled = StandardScaler().fit_transform(df[numeric].dropna())

    inertias = [KMeans(k, n_init="auto", random_state=RANDOM_STATE)
                .fit(X_scaled).inertia_ for k in range(1, 11)]
    st.plotly_chart(
        px.line(x=list(range(1, 11)), y=inertias, markers=True,
                labels={"x": "k", "y": "Inertia"},
                title="Elbow Method"),
        use_container_width=True,
    )

    k = st.slider("Choose k for clustering", 2, 10, 3)
    km = KMeans(k, n_init="auto", random_state=RANDOM_STATE).fit(X_scaled)
    df["cluster"] = km.labels_

    persona = df.groupby("cluster").agg(
        {c: ("mean" if c in numeric else "first") for c in df.columns}
    )
    st.subheader("Cluster Personas")
    st.dataframe(persona, use_container_width=True)

    st.markdown(download_link(df, "clustered_data.csv",
                              "📥 Download dataset with clusters"),
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Tab 4 – Association Rule Mining
# ──────────────────────────────────────────────────────────────────────
def tab_association(df: pd.DataFrame):
    st.header("🔗 Association Rule Mining")
    cols = st.multiselect("Select exactly 3 categorical columns",
                          df.columns.tolist(),
                          default=["JobRole", "MaritalStatus", "OverTime"])
    if len(cols) != 3:
        st.warning("Please select **exactly** three columns.")
        return

    min_sup  = st.slider("min_support",     0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("min_confidence",  0.01, 1.0, 0.30, 0.01)
    min_lift = st.slider("min_lift",        0.50, 5.0, 1.00, 0.10)

    hot = pd.get_dummies(df[cols].astype(str))
    rules = association_rules(apriori(hot, min_support=min_sup, use_colnames=True),
                              metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift]\
            .sort_values("confidence", ascending=False).head(10)

    st.dataframe(rules[["antecedents", "consequents", "support",
                        "confidence", "lift"]])
    st.plotly_chart(
        px.bar(rules, x=rules.index.astype(str), y="lift",
               title="Lift of Top-10 Rules"),
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────
# Tab 5 – Regression Insights
# ──────────────────────────────────────────────────────────────────────
def tab_regression(df: pd.DataFrame):
    st.header("📈 Regression Insights")
    target = st.selectbox("Target variable",
                          df.select_dtypes("number").columns,
                          index=df.columns.get_loc(DEFAULT_REG_TARGET))
    y = df[target]
    X = df.drop(columns=[target])
    numeric, categorical = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=6,
                                               random_state=RANDOM_STATE),
    }
    scores = []
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", model)]).fit(X, y)
        r2 = pipe.score(X, y)
        scores.append((name, round(r2, 3)))

        if name in ("Linear", "Ridge", "Lasso"):
            coefs = pd.DataFrame({
                "feature": pipe["prep"].get_feature_names_out(),
                "coef": model.coef_,
            }).sort_values("coef", key=np.abs, ascending=False).head(10)
            with st.expander(f"{name} – Top Coefficients"):
                st.plotly_chart(
                    px.bar(coefs, x="coef", y="feature", orientation="h"),
                    use_container_width=True,
                )

    st.subheader("Model R² Scores")
    st.table(pd.DataFrame(scores, columns=["Model", "R²"])
             .set_index("Model"))

    dt_pipe = Pipeline([("prep", pre), ("mdl", models["Decision Tree"])])
    dt_pipe.fit(X, y)
    preds = dt_pipe.predict(X)
    resid = y - preds
    st.plotly_chart(
        px.scatter(x=preds, y=resid,
                   labels={"x": "Predicted", "y": "Residual"},
                   title="Residuals – Decision Tree"),
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────
# Tab 6 – 12-Month Retention Forecast
# ──────────────────────────────────────────────────────────────────────
def tab_retention(df: pd.DataFrame):
    st.header("⏳ 12-Month Retention Forecast")
    alg = st.selectbox("Algorithm",
                       ["Logistic Regression", "Random Forest", "XGBoost"])
    horizon = st.slider("Horizon (months)", 6, 24, 12)

    if "YearsAtCompany" not in df.columns:
        st.error("Column `YearsAtCompany` not found.")
        return

    tmp = df.copy()
    tmp["Stay"] = (tmp["YearsAtCompany"] * 12 >= horizon).astype(int)

    X = tmp.drop(columns=["Stay"])
    y = tmp["Stay"]
    numeric, categorical = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
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

    pipe = Pipeline([("prep", pre), ("mdl", mdl)]).fit(X, y)
    tmp["RetentionProb"] = pipe.predict_proba(X)[:, 1]
    st.dataframe(tmp[["EmployeeNumber", "RetentionProb"]].head(),
                 use_container_width=True)

    st.subheader("Feature Importance")
    explainer = shap.Explainer(pipe["mdl"])
    shap_values = explainer(pipe["prep"].transform(X)[:200])
    imp = pd.DataFrame({
        "feature": pipe["prep"].get_feature_names_out(),
        "importance": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("importance", ascending=False).head(15)
    st.plotly_chart(
        px.bar(imp, x="importance", y="feature", orientation="h"),
        use_container_width=True,
    )

    st.markdown(
        download_link(tmp[["EmployeeNumber", "RetentionProb"]],
                      "retention_predictions.csv",
                      "📥 Download predictions"),
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("📂 Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload HR Analytics CSV",
                                             type="csv")
    df = universal_filters(load_data(uploaded_file))

    st.sidebar.download_button("Download filtered CSV",
                               df.to_csv(index=False).encode(),
                               "filtered_data.csv",
                               "text/csv")

    tabs = st.tabs([
        "Data Visualisation",
        "Classification",
        "Clustering",
        "Association Rules",
        "Regression",
        "12-Month Forecast",
    ])

    with tabs[0]:
        tab_visualisation(df)
    with tabs[1]:
        tab_classification(df)
    with tabs[2]:
        tab_clustering(df)
    with tabs[3]:
        tab_association(df)
    with tabs[4]:
        tab_regression(df)
    with tabs[5]:
        tab_retention(df)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
