# HR Analytics & Retention Dashboard (Streamlit Cloud)

This multipage Streamlit app turns **HR Analytics.csv** into a production-ready talent-management cockpit:

| Page | What you’ll find |
|------|------------------|
| 📊 Data Visualisation | 20+ auto-generated descriptive insights |
| 🤖 Classification     | Four-model attrition predictor + ROC, confusion matrix & batch scoring |
| 🔍 Clustering         | K-means personas with elbow chart & downloadable clusters |
| 🔗 Association Rules  | Apriori mining on any 3 categorical columns |
| 📈 Regression Insights| Linear, Ridge, Lasso & Tree regressions with coefficient/feature plots |
| ⏳ Retention Forecast | 12-month stay probability + feature-importance chart |

## 1 - Quick Start (Local)

```bash
git clone https://github.com/<you>/hr-analytics-dashboard.git
cd hr-analytics-dashboard
pip install -r requirements.txt
streamlit run app.py
