#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Analytics Streamlit Dashboard
Author  : Jaideep (GMBA) — aided by ChatGPT o3
Updated : 2025-07-03  (slider fix + metrics-table fix)
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="HR Analytics Dashboard",
                   page_icon=":bar_chart:", layout="wide")

# ---------------------------------------------------------------------
DATA_PATH = Path(__file__).with_name("HR Analytics.csv")
TARGET_CLASSIFICATION = "Attrition"
DEFAULT_REG_TARGET = "MonthlyIncome"
RANDOM_STATE = 42
# ---------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)
    if TARGET_CLASSIFICATION in df.columns:
        df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].astype("category")
    return df
# ---------------------------------------------------------------------
def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = df.select_dtypes("number").columns.tolist()
    cat = [c for c in df.columns if c not in num]
    return num, cat
# ---------------------------------------------------------------------
def universal_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Universal Filters")
    num, cat = get_column_types(df)
    df_f = df.copy()

    with st.sidebar.expander("Numeric Ranges"):
        for col in num:
            lo, hi = df[col].min(), df[col].max()
            if lo == hi:
                st.number_input(col, value=float(lo),
                                disabled=True,
                                help=f"{col} constant ({lo})")
                continue
            sel_lo, sel_hi = st.slider(col, float(lo), float(hi),
                                       (float(lo), float(hi)),
                                       help=f"Filter {col}")
            df_f = df_f[(df_f[col] >= sel_lo) & (df_f[col] <= sel_hi)]

    with st.sidebar.expander("Categorical Values"):
        for col in cat:
            opts = df[col].dropna().unique().tolist()
            sel = st.multiselect(col, opts, default=opts)
            df_f = df_f[df_f[col].isin(sel)]
    return df_f
# ---------------------------------------------------------------------
def download_link(obj, name, label):
    txt = obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else obj
    b64 = base64.b64encode(txt.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{name}">{label}</a>'
# ---------------------------------------------------------------------
def tab_visualisation(df):
    st.header("📊 Data Visualisation")
    num, cat = get_column_types(df)
    with st.expander("Attrition % by Department"):
        cross = pd.crosstab(df["Department"], df[TARGET_CLASSIFICATION],
                            normalize="index") * 100
        st.plotly_chart(px.bar(cross.reset_index(), x="Department", y="Yes",
                               labels={"Yes": "Attrition %"}),
                        use_container_width=True)
    with st.expander("Age Distribution"):
        st.plotly_chart(px.histogram(df, x="Age", color=TARGET_CLASSIFICATION,
                                     nbins=30, barmode="overlay"),
                        use_container_width=True)
    with st.expander("Income vs Job Level"):
        st.plotly_chart(px.violin(df, x="JobLevel", y="MonthlyIncome",
                                  color=TARGET_CLASSIFICATION,
                                  box=True, points="outliers"),
                        use_container_width=True)
    with st.expander("Correlation Heat-map"):
        st.plotly_chart(px.imshow(df[num].corr(), text_auto=".2f"),
                        use_container_width=True)
    auto = 4
    for col in cat[:8]:
        with st.expander(f"Countplot – {col}"):
            st.plotly_chart(px.histogram(df, x=col,
                                         color=TARGET_CLASSIFICATION,
                                         barmode="group"),
                            use_container_width=True)
            auto += 1
    for col in num[:8]:
        with st.expander(f"Boxplot – {col}"):
            st.plotly_chart(px.box(df, y=col, color=TARGET_CLASSIFICATION),
                            use_container_width=True)
            auto += 1
    st.success(f"Rendered **{auto}** visual insights.")
# ---------------------------------------------------------------------
def preprocess_for_modeling(df, target):
    num, cat = get_column_types(df.drop(columns=[target]))
    X = df.drop(columns=[target])
    y = df[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return X, y, pre
# ---------------------------------------------------------------------
def train_classifiers(X, y, pre):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                                                random_state=RANDOM_STATE),
        "GBRT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    out: Dict[str, Dict] = {}
    for n, m in models.items():
        p = Pipeline([("prep", pre), ("mdl", m)]).fit(X_tr, y_tr)
        out[n] = {
            "pipe": p,
            "train": {k: v(y_tr, p.predict(X_tr), pos_label="Yes")
                      for k, v in
                      [("accuracy", accuracy_score),
                       ("precision", precision_score),
                       ("recall", recall_score),
                       ("f1", f1_score)]},
            "test": {k: v(y_te, p.predict(X_te), pos_label="Yes")
                     for k, v in
                     [("accuracy", accuracy_score),
                      ("precision", precision_score),
                      ("recall", recall_score),
                      ("f1", f1_score)]},
            "proba": p.predict_proba(X_te)[:, 1],
            "y_test": y_te,
        }
    return out
# ---------------------------------------------------------------------
def tab_classification(df):
    st.header("🤖 Classification")
    X, y, pre = preprocess_for_modeling(df, TARGET_CLASSIFICATION)
    res = train_classifiers(X, y, pre)

    # ----- fixed metrics-table block ----------------------------------
    rows = []
    for n, r in res.items():
        rows.append([
            n,
            *(round(r["train"][m], 3) for m in ("accuracy", "precision", "recall", "f1")),
            *(round(r["test"][m], 3)  for m in ("accuracy", "precision", "recall", "f1")),
        ])
    cols_nested = pd.MultiIndex.from_product(
        [["Train", "Test"], ["Accuracy", "Precision", "Recall", "F1"]]
    )
    metrics_df = pd.DataFrame(rows, columns=["Model"] + list(cols_nested))\
                 .set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)
    # ------------------------------------------------------------------

    model_sel = st.selectbox("Confusion matrix model", res.keys())
    y_true = res[model_sel]["y_test"]
    y_pred = res[model_sel]["pipe"].predict(X.iloc[y_true.index])
    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["No", "Yes"],
                              y=["No", "Yes"],
                              labels=dict(x="Predicted", y="Actual")),
                    use_container_width=False)

    roc = go.Figure()
    for n, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_test"].map({"No": 0, "Yes": 1}),
                                r["proba"])
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n))
    roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                  line=dict(dash="dash"))
    roc.update_layout(title="ROC Curves",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    st.plotly_chart(roc, use_container_width=True)

    st.subheader("🔮 Batch prediction")
    up = st.file_uploader("CSV without *Attrition*", type="csv")
    if up:
        new_df = pd.read_csv(up)
        best = max(res.items(),
                   key=lambda kv: kv[1]["test"]["f1"])[1]["pipe"]
        new_df["PredictedAttrition"] = best.predict(new_df)
        st.dataframe(new_df.head(), use_container_width=True)
        st.download_button("Download predictions",
                           new_df.to_csv(index=False).encode(),
                           "predictions.csv")
# ---------------------------------------------------------------------
def tab_clustering(df):
    st.header("🕵️‍♀️ Clustering")
    num, _ = get_column_types(df)
    X_s = StandardScaler().fit_transform(df[num].dropna())
    inertias = [KMeans(k, n_init="auto", random_state=RANDOM_STATE)
                .fit(X_s).inertia_ for k in range(1, 11)]
    st.plotly_chart(px.line(x=list(range(1, 11)), y=inertias,
                            markers=True,
                            labels={"x": "k", "y": "Inertia"}),
                    use_container_width=True)
    k = st.slider("Choose k", 2, 10, 3)
    df["cluster"] = KMeans(k, n_init="auto",
                           random_state=RANDOM_STATE).fit_predict(X_s)
    persona = df.groupby("cluster").agg(
        {c: ("mean" if c in num else "first") for c in df})
    st.dataframe(persona, use_container_width=True)
    st.markdown(download_link(df, "clustered.csv",
                              "📥 Download clusters"),
                unsafe_allow_html=True)
# ---------------------------------------------------------------------
def tab_association(df):
    st.header("🔗 Association Rule Mining")
    cols = st.multiselect("Select three categorical columns",
                          df.columns.tolist(),
                          default=["JobRole", "MaritalStatus", "OverTime"])
    if len(cols) != 3:
        st.warning("Exactly three columns required.")
        return
    sup = st.slider("min_support", 0.01, 0.5, 0.05, 0.01)
    conf = st.slider("min_confidence", 0.01, 1.0, 0.3, 0.01)
    lift = st.slider("min_lift", 0.5, 5.0, 1.0, 0.1)
    hot = pd.get_dummies(df[cols].astype(str))
    rules = association_rules(apriori(hot, min_support=sup,
                                      use_colnames=True),
                              metric="confidence",
                              min_threshold=conf)
    rules = rules[rules["lift"] >= lift].sort_values("confidence",
                                                     ascending=False).head(10)
    st.dataframe(rules[["antecedents", "consequents",
                        "support", "confidence", "lift"]])
    st.plotly_chart(px.bar(rules, x=rules.index.astype(str), y="lift"),
                    use_container_width=True)
# ---------------------------------------------------------------------
def tab_regression(df):
    st.header("📈 Regression Insights")
    target = st.selectbox("Target variable",
                          df.select_dtypes("number").columns,
                          index=df.columns.get_loc(DEFAULT_REG_TARGET))
    y, X = df[target], df.drop(columns=[target])
    num, cat = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    models = {"Linear": LinearRegression(), "Ridge": Ridge(),
              "Lasso": Lasso(alpha=0.01),
              "Decision Tree": DecisionTreeRegressor(max_depth=6,
                                                      random_state=RANDOM_STATE)}
    r2s = []
    for n, m in models.items():
        p = Pipeline([("prep", pre), ("mdl", m)]).fit(X, y)
        r2 = p.score(X, y)
        r2s.append((n, round(r2, 3)))
        if n in ("Linear", "Ridge", "Lasso"):
            co = pd.DataFrame({"f": p["prep"].get_feature_names_out(),
                               "c": m.coef_})\
                 .sort_values("c", key=np.abs, ascending=False).head(10)
            with st.expander(f"{n} Coefficients"):
                st.plotly_chart(px.bar(co, x="c", y="f",
                                       orientation="h"),
                                use_container_width=True)
    st.table(pd.DataFrame(r2s, columns=["Model", "R²"])
             .set_index("Model"))
    dt_pred = models["Decision Tree"].fit(
        pre.fit_transform(X), y).predict(pre.transform(X))
    st.plotly_chart(px.scatter(x=dt_pred, y=y-dt_pred,
                               labels={"x": "Predicted", "y": "Residual"}),
                    use_container_width=True)
# ---------------------------------------------------------------------
def tab_retention(df):
    st.header("⏳ 12-Month Retention Forecast")
    alg = st.selectbox("Algorithm", ["Logistic Regression",
                                     "Random Forest", "XGBoost"])
    horizon = st.slider("Horizon (months)", 6, 24, 12)
    if "YearsAtCompany" not in df:
        st.error("YearsAtCompany missing.")
        return
    tmp = df.copy()
    tmp["Stay"] = (tmp["YearsAtCompany"]*12 >= horizon).astype(int)
    X, y = tmp.drop(columns=["Stay"]), tmp["Stay"]
    num, cat = get_column_types(X)
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    mdl = (LogisticRegression(max_iter=1000) if alg == "Logistic Regression"
           else RandomForestClassifier(n_estimators=400,
                                       random_state=RANDOM_STATE)
           if alg == "Random Forest"
           else xgb.XGBClassifier(random_state=RANDOM_STATE,
                                  eval_metric="logloss",
                                  learning_rate=0.05,
                                  n_estimators=500, max_depth=5))
    pipe = Pipeline([("prep", pre), ("mdl", mdl)]).fit(X, y)
    tmp["RetentionProb"] = pipe.predict_proba(X)[:, 1]
    st.dataframe(tmp[["EmployeeNumber", "RetentionProb"]].head(),
                 use_container_width=True)
    expl = shap.Explainer(pipe["mdl"])
    sv = expl(pipe["prep"].transform(X)[:200])
    imp = pd.DataFrame({"feat": pipe["prep"].get_feature_names_out(),
                        "imp": np.abs(sv.values).mean(axis=0)})\
          .sort_values("imp", ascending=False).head(15)
    st.plotly_chart(px.bar(imp, x="imp", y="feat", orientation="h"),
                    use_container_width=True)
    st.markdown(download_link(tmp[["EmployeeNumber", "RetentionProb"]],
                              "retention.csv",
                              "📥 Download predictions"),
                unsafe_allow_html=True)
# ---------------------------------------------------------------------
def main():
    st.sidebar.title("📂 Data Source")
    up = st.sidebar.file_uploader("Upload HR Analytics CSV", type="csv")
    df = universal_filters(load_data(up))
    st.sidebar.download_button("Download filtered CSV",
                               df.to_csv(index=False).encode(),
                               "filtered_data.csv")
    tabs = st.tabs(["Data Visualisation", "Classification", "Clustering",
                    "Association Rules", "Regression",
                    "12-Month Forecast"])
    tab_visualisation(df); tab_classification(df); tab_clustering(df)
    tab_association(df); tab_regression(df); tab_retention(df)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
