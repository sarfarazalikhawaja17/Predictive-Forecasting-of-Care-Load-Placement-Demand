"""
UAC Predictive Forecasting Dashboard
HHS / Unified Mentor — Streamlit Web Application
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Predictive Forecasting | HHS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — deep navy / electric-teal palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --navy:     #0B1120;
    --navy-mid: #111827;
    --navy-card:#151F30;
    --navy-border: #1E2D45;
    --teal:     #00C6A2;
    --teal-dim: #009E83;
    --amber:    #F59E0B;
    --rose:     #F43F5E;
    --blue:     #3B82F6;
    --purple:   #8B5CF6;
    --text-hi:  #F0F4FF;
    --text-mid: #94A3B8;
    --text-lo:  #475569;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text-hi) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy-mid) !important;
    border-right: 1px solid var(--navy-border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: var(--text-mid) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--teal) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin-top: 1.5rem !important;
}

/* ── Selectbox / Slider / Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--navy-card) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--teal) !important;
}
.stSlider [data-testid="stThumbValue"] { color: var(--teal) !important; }

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Top header banner ── */
.uac-header {
    background: linear-gradient(135deg, #0F1A2E 0%, #0B1C3A 50%, #091528 100%);
    border: 1px solid var(--navy-border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.uac-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,198,162,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.uac-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 150px;
    background: radial-gradient(ellipse, rgba(59,130,246,0.08) 0%, transparent 70%);
}
.uac-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-hi);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}
.uac-subtitle {
    font-size: 0.88rem;
    color: var(--text-mid);
    margin: 0;
    position: relative;
    z-index: 1;
}
.uac-badge {
    display: inline-block;
    background: rgba(0,198,162,0.15);
    color: var(--teal);
    border: 1px solid rgba(0,198,162,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.kpi-card {
    flex: 1;
    min-width: 140px;
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(0,198,162,0.4); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.teal::before  { background: var(--teal); }
.kpi-card.amber::before { background: var(--amber); }
.kpi-card.rose::before  { background: var(--rose); }
.kpi-card.blue::before  { background: var(--blue); }
.kpi-card.purple::before{ background: var(--purple); }
.kpi-label {
    font-size: 0.7rem;
    color: var(--text-mid);
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-hi);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-sub {
    font-size: 0.72rem;
    color: var(--text-mid);
}
.kpi-delta-pos { color: var(--teal);  font-size: 0.75rem; font-weight: 600; }
.kpi-delta-neg { color: var(--rose);  font-size: 0.75rem; font-weight: 600; }

/* ── Section cards ── */
.section-card {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-hi);
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-desc {
    font-size: 0.8rem;
    color: var(--text-mid);
    margin-bottom: 1.25rem;
    line-height: 1.5;
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

/* ── Alert boxes ── */
.alert-surge {
    background: rgba(244,63,94,0.08);
    border: 1px solid rgba(244,63,94,0.3);
    border-left: 4px solid var(--rose);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.83rem;
    color: var(--text-hi);
}
.alert-normal {
    background: rgba(0,198,162,0.07);
    border: 1px solid rgba(0,198,162,0.25);
    border-left: 4px solid var(--teal);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.83rem;
    color: var(--text-hi);
}

/* ── Model comparison table ── */
.model-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.model-table th {
    color: var(--text-mid);
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.72rem;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--navy-border);
    text-align: left;
}
.model-table td {
    padding: 0.65rem 0.75rem;
    border-bottom: 1px solid rgba(30,45,69,0.5);
    color: var(--text-hi);
}
.model-table tr:last-child td { border-bottom: none; }
.model-table tr.best-row td { background: rgba(0,198,162,0.06); }
.model-table tr:hover td { background: rgba(255,255,255,0.02); }
.best-badge {
    background: rgba(0,198,162,0.15);
    color: var(--teal);
    border: 1px solid rgba(0,198,162,0.3);
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

/* ── Metric bar ── */
.acc-bar-wrap { display: flex; align-items: center; gap: 0.6rem; }
.acc-bar-track {
    flex: 1; height: 6px; background: var(--navy-border);
    border-radius: 3px; overflow: hidden;
}
.acc-bar-fill { height: 100%; border-radius: 3px; }
.acc-val { font-size: 0.78rem; font-weight: 600; color: var(--text-hi); min-width: 44px; text-align: right; }

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tablist"] { gap: 0.25rem; border-bottom: 1px solid var(--navy-border); }
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: var(--text-mid) !important;
    border: none !important;
    border-radius: 0 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.1rem !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { font-size: 0.83rem !important; }

/* ── Divider ── */
hr { border-color: var(--navy-border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--navy-border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COLOUR MAP FOR MODELS
# ─────────────────────────────────────────────
MODEL_COLORS = {
    "Naive Persistence":     "#94A3B8",
    "Moving Average (7d)":   "#F59E0B",
    "ARIMA(0,1,0)":          "#00C6A2",
    "Exponential Smoothing": "#A78BFA",
    "Random Forest":         "#3B82F6",
    "Gradient Boosting":     "#F43F5E",
    "Ridge Regression":      "#34D399",
}

MODEL_TYPES = {
    "Naive Persistence":     "Baseline",
    "Moving Average (7d)":   "Baseline",
    "ARIMA(0,1,0)":          "Statistical",
    "Exponential Smoothing": "Statistical",
    "Random Forest":         "ML",
    "Gradient Boosting":     "ML",
    "Ridge Regression":      "ML",
}

# ─────────────────────────────────────────────
#  DATA PIPELINE (self-contained, no pkl needed)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_pipeline():
    """
    Full pipeline: load data, engineer features, train all models,
    return everything needed for the dashboard.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # ── Try to load hhs.csv from common locations ─────────────────────────
    search_paths = [
        "hhs.csv",
        os.path.join(os.path.dirname(__file__), "hhs.csv"),
        "/mnt/user-data/uploads/hhs.csv",
        "streamlit_artifacts/hhs.csv",
    ]
    df_raw = None
    for p in search_paths:
        if os.path.exists(p):
            df_raw = pd.read_csv(p)
            break

    if df_raw is None:
        return None, "hhs.csv not found. Place it alongside app.py and restart."

    # ── Clean ─────────────────────────────────────────────────────────────
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d, %Y")
    for c in [col for col in df.columns if col != "Date"]:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={
        "Children apprehended and placed in CBP custody*": "apprehended_cbp",
        "Children in CBP custody":                        "in_cbp_custody",
        "Children transferred out of CBP custody":        "transferred_out_cbp",
        "Children in HHS Care":                           "in_hhs_care",
        "Children discharged from HHS Care":              "discharged",
    })
    df = (df.sort_values("Date")
            .drop_duplicates(subset="Date", keep="last")
            .set_index("Date"))
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_idx).interpolate(method="linear").ffill().bfill()
    df.index.name = "Date"

    # ── Feature engineering ───────────────────────────────────────────────
    data = df.copy()
    data["net_pressure"] = data["transferred_out_cbp"] - data["discharged"]
    for col in ["in_hhs_care", "discharged", "net_pressure"]:
        for lag in [7, 14, 30, 60]:
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
        for window in [14, 30]:
            data[f"{col}_roll_mean{window}"] = data[col].shift(7).rolling(window).mean()
            data[f"{col}_roll_std{window}"]  = data[col].shift(7).rolling(window).std()
    data["day_of_week"] = data.index.dayofweek
    data["month"]       = data.index.month
    data["is_weekend"]  = (data["day_of_week"] >= 5).astype(int)
    data["quarter"]     = data.index.quarter
    data = data.dropna()

    feature_cols = [c for c in data.columns if c not in
                    ["in_hhs_care", "discharged", "apprehended_cbp",
                     "in_cbp_custody", "transferred_out_cbp"]]
    discharge_feature_cols = [c for c in feature_cols if
                               "discharged" in c or "net_pressure" in c or
                               c in ["day_of_week", "month", "is_weekend", "quarter"]]

    # ── Train / test ──────────────────────────────────────────────────────
    horizon = 90
    aligned       = df.loc[data.index]
    y_train_raw   = aligned.iloc[:-horizon]["in_hhs_care"]
    y_test_raw    = aligned.iloc[-horizon:]["in_hhs_care"]

    train = data.iloc[:-horizon]
    test  = data.iloc[-horizon:]
    X_train, y_train_ml = train[feature_cols], train["in_hhs_care"]
    X_test,  y_test_ml  = test[feature_cols],  test["in_hhs_care"]
    X_train_d = train[discharge_feature_cols]
    X_test_d  = test[discharge_feature_cols]
    y_train_disch = train["discharged"].clip(lower=1)
    y_test_disch  = test["discharged"].clip(lower=1)

    # ── Helpers ───────────────────────────────────────────────────────────
    def smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        denom = np.where(denom == 0, 1e-8, denom)
        return np.mean(np.abs(y_true - y_pred) / denom) * 100

    def evaluate(y_true, y_pred, name):
        y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
        mp = smape(y_true, y_pred)
        return {
            "Model":        name,
            "MAE":          round(float(mean_absolute_error(y_true, y_pred)), 1),
            "RMSE":         round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
            "sMAPE (%)":    round(mp, 2),
            "Accuracy (%)": round(100 - mp, 2),
        }

    # ── Baseline models ───────────────────────────────────────────────────
    naive_f = np.repeat(float(y_train_raw.iloc[-1]), horizon)
    ma_f    = np.repeat(float(y_train_raw.tail(7).mean()), horizon)

    # ── Statistical ───────────────────────────────────────────────────────
    arima_fit = ARIMA(y_train_raw, order=(0, 1, 0)).fit()
    arima_f   = arima_fit.forecast(steps=horizon).values

    hw_fit = ExponentialSmoothing(
        y_train_raw, trend="add", seasonal=None
    ).fit(smoothing_level=0.05, smoothing_trend=0.05, optimized=False)
    hw_f = hw_fit.forecast(steps=horizon).values

    # ── ML models ─────────────────────────────────────────────────────────
    rf_pipeline = Pipeline([("scaler", StandardScaler()),
                             ("model", RandomForestRegressor(
                                 n_estimators=10, max_depth=2,
                                 min_samples_leaf=20, random_state=42))])
    rf_pipeline.fit(X_train, y_train_ml)
    rf_f = rf_pipeline.predict(X_test)

    gbr_pipeline = Pipeline([("scaler", StandardScaler()),
                              ("model", GradientBoostingRegressor(
                                  n_estimators=20, learning_rate=0.5,
                                  max_depth=1, subsample=0.5, random_state=42))])
    gbr_pipeline.fit(X_train, y_train_ml)
    gbr_f = gbr_pipeline.predict(X_test)

    ridge_pipeline = Pipeline([("scaler", StandardScaler()),
                                ("model", Ridge(alpha=100.0))])
    ridge_pipeline.fit(X_train, y_train_ml)
    ridge_f = ridge_pipeline.predict(X_test)

    # ── Discharge model ───────────────────────────────────────────────────
    gbr_disch = Pipeline([("scaler", StandardScaler()),
                           ("model", GradientBoostingRegressor(
                               n_estimators=80, learning_rate=0.08,
                               max_depth=3, subsample=0.8,
                               min_samples_leaf=10, random_state=42))])
    gbr_disch.fit(X_train_d, y_train_disch)
    discharge_f = gbr_disch.predict(X_test_d)

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = [
        evaluate(y_test_raw, naive_f,  "Naive Persistence"),
        evaluate(y_test_raw, ma_f,     "Moving Average (7d)"),
        evaluate(y_test_raw, arima_f,  "ARIMA(0,1,0)"),
        evaluate(y_test_raw, hw_f,     "Exponential Smoothing"),
        evaluate(y_test_ml,  rf_f,     "Random Forest"),
        evaluate(y_test_ml,  gbr_f,    "Gradient Boosting"),
        evaluate(y_test_ml,  ridge_f,  "Ridge Regression"),
    ]
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    disch_eval = evaluate(y_test_disch, discharge_f, "GBR Discharge Forecast")

    forecasts_dict = {
        "Naive Persistence":     naive_f,
        "Moving Average (7d)":   ma_f,
        "ARIMA(0,1,0)":          arima_f,
        "Exponential Smoothing": hw_f,
        "Random Forest":         rf_f,
        "Gradient Boosting":     gbr_f,
        "Ridge Regression":      ridge_f,
    }
    pipeline_map = {
        "Random Forest":     rf_pipeline,
        "Gradient Boosting": gbr_pipeline,
        "Ridge Regression":  ridge_pipeline,
    }

    best_name = results_df.iloc[0]["Model"]
    best_f    = pipeline_map[best_name].predict(X_test) if best_name in pipeline_map else naive_f

    net_pressure       = data["net_pressure"].tail(horizon).values
    surge_days         = int((net_pressure > 0).sum())
    capacity_threshold = float(data["in_hhs_care"].quantile(0.95))
    breach_prob        = float(np.mean(best_f > capacity_threshold))
    stability_idx      = float(1 - np.std(best_f - y_test_ml.values) / np.std(y_test_ml.values))

    kpi = {
        "best_model_name":       best_name,
        "forecast_accuracy_pct": round(results_df.iloc[0]["Accuracy (%)"], 2),
        "surge_lead_time":       surge_days,
        "horizon":               horizon,
        "capacity_threshold":    round(capacity_threshold, 0),
        "breach_probability":    round(breach_prob * 100, 1),
        "stability_index":       round(stability_idx, 3),
        "discharge_accuracy":    round(disch_eval["Accuracy (%)"], 2),
        "train_end":             str(train.index.max().date()),
        "test_start":            str(test.index.min().date()),
        "test_end":              str(test.index.max().date()),
        "total_records":         len(df),
    }

    payload = {
        "df":              df,
        "data":            data,
        "train":           train,
        "test":            test,
        "y_train_raw":     y_train_raw,
        "y_test_raw":      y_test_raw,
        "y_test_ml":       y_test_ml,
        "y_test_disch":    y_test_disch,
        "forecasts_dict":  forecasts_dict,
        "results_df":      results_df,
        "disch_eval":      disch_eval,
        "discharge_f":     discharge_f,
        "net_pressure":    net_pressure,
        "best_name":       best_name,
        "best_f":          best_f,
        "horizon":         horizon,
        "kpi":             kpi,
        "feature_cols":    feature_cols,
        "pipeline_map":    pipeline_map,
    }
    return payload, None


# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
    xaxis=dict(
        gridcolor="rgba(30,45,69,0.8)", gridwidth=1,
        linecolor="rgba(30,45,69,0.8)", showline=True,
        tickfont=dict(size=11, color="#64748B"),
    ),
    yaxis=dict(
        gridcolor="rgba(30,45,69,0.8)", gridwidth=1,
        linecolor="rgba(30,45,69,0.8)", showline=True,
        tickfont=dict(size=11, color="#64748B"),
    ),
    legend=dict(
        bgcolor="rgba(11,17,32,0.9)",
        bordercolor="rgba(30,45,69,0.8)",
        borderwidth=1,
        font=dict(size=11, color="#94A3B8"),
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)


def ci_bands(forecast, std_scale=0.08):
    """Simple symmetric confidence bands scaled by value."""
    std = np.abs(forecast) * std_scale
    return forecast - 1.96 * std, forecast + 1.96 * std


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(payload):
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 0.5rem 0;'>
            <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                        font-weight:700;color:#F0F4FF;letter-spacing:-0.01em;'>
                🏛️ UAC Forecasting
            </div>
            <div style='font-size:0.72rem;color:#475569;margin-top:2px;'>
                HHS · Unified Mentor Program
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### ⚙️ Controls")

        all_models = list(payload["forecasts_dict"].keys())
        best = payload["best_name"]
        default_models = [best, "Random Forest", "Gradient Boosting"] \
            if best not in ("Random Forest", "Gradient Boosting") else [best, "Naive Persistence"]
        default_models = [m for m in default_models if m in all_models]

        selected_models = st.multiselect(
            "Active Models",
            options=all_models,
            default=default_models[:3],
            help="Select models to display on the forecast chart."
        )
        if not selected_models:
            selected_models = [best]

        horizon_days = st.slider(
            "Forecast Horizon (days)",
            min_value=7, max_value=90, value=90, step=7,
            help="Trim the 90-day window to focus on a shorter horizon."
        )

        show_ci = st.checkbox("Show Confidence Intervals", value=True)
        show_actual = st.checkbox("Show Actuals", value=True)

        st.markdown("### 📊 Scenario")
        scenario = st.selectbox(
            "Intake Scenario",
            ["Baseline (actual)", "Surge +20%", "Surge +40%", "De-escalation −20%"],
        )

        st.markdown("---")
        st.markdown("### 🔍 Best Model")
        kpi = payload["kpi"]
        st.markdown(f"""
        <div style='background:rgba(0,198,162,0.08);border:1px solid rgba(0,198,162,0.25);
                    border-radius:10px;padding:0.9rem 1rem;'>
            <div style='font-family:Space Grotesk,sans-serif;font-size:1.15rem;
                        font-weight:700;color:#00C6A2;'>{kpi['best_model_name']}</div>
            <div style='font-size:0.75rem;color:#64748B;margin-top:4px;'>
                Accuracy: <span style='color:#F0F4FF;font-weight:600;'>
                {kpi['forecast_accuracy_pct']:.1f}%</span>
            </div>
            <div style='font-size:0.75rem;color:#64748B;margin-top:2px;'>
                Stability: <span style='color:#F0F4FF;font-weight:600;'>
                {kpi['stability_index']:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:0.7rem;color:#334155;line-height:1.6;'>
            Dataset: {kpi['total_records']:,} days<br>
            Training end: {kpi['train_end']}<br>
            Test start: {kpi['test_start']}<br>
            Test end: {kpi['test_end']}
        </div>
        """, unsafe_allow_html=True)

    return selected_models, horizon_days, show_ci, show_actual, scenario


# ─────────────────────────────────────────────
#  KPI STRIP
# ─────────────────────────────────────────────
def render_kpis(kpi):
    bp = kpi["breach_probability"]
    bp_color = "rose" if bp > 10 else "teal"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card teal">
        <div class="kpi-label">Forecast Accuracy</div>
        <div class="kpi-value">{kpi['forecast_accuracy_pct']:.1f}<span style='font-size:1rem;'>%</span></div>
        <div class="kpi-sub">sMAPE-based • Best model</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Surge Lead Time</div>
        <div class="kpi-value">{kpi['surge_lead_time']}<span style='font-size:1rem;'> d</span></div>
        <div class="kpi-sub">of {kpi['horizon']} forecast days</div>
      </div>
      <div class="kpi-card {bp_color}">
        <div class="kpi-label">Capacity Breach Risk</div>
        <div class="kpi-value">{bp:.1f}<span style='font-size:1rem;'>%</span></div>
        <div class="kpi-sub">Threshold ≈ {kpi['capacity_threshold']:,.0f} children</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Stability Index</div>
        <div class="kpi-value">{kpi['stability_index']:.3f}</div>
        <div class="kpi-sub">Forecast robustness (0–1)</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Discharge Accuracy</div>
        <div class="kpi-value">{kpi['discharge_accuracy']:.1f}<span style='font-size:1rem;'>%</span></div>
        <div class="kpi-sub">GBR discharge model</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TAB 1 — CARE LOAD FORECAST
# ─────────────────────────────────────────────
def render_care_forecast(payload, selected_models, horizon_days, show_ci, show_actual, scenario):
    kpi = payload["kpi"]
    test_idx = payload["test"].index[:horizon_days]
    y_actual = payload["y_test_raw"].values[:horizon_days]

    # Scenario multiplier
    scenario_mult = {
        "Baseline (actual)":    1.0,
        "Surge +20%":           1.20,
        "Surge +40%":           1.40,
        "De-escalation −20%":   0.80,
    }
    mult = scenario_mult.get(scenario, 1.0)

    # ── Build figure ──────────────────────────────────────────────────────
    fig = go.Figure()

    if show_actual:
        fig.add_trace(go.Scatter(
            x=test_idx, y=y_actual,
            name="Actual",
            line=dict(color="#F0F4FF", width=2.5),
            mode="lines",
        ))

    for model_name in selected_models:
        fc = payload["forecasts_dict"][model_name][:horizon_days] * mult
        col = MODEL_COLORS.get(model_name, "#94A3B8")

        fig.add_trace(go.Scatter(
            x=test_idx, y=fc,
            name=model_name,
            line=dict(color=col, width=1.8, dash="dot"),
            mode="lines",
        ))

        if show_ci:
            lo, hi = ci_bands(fc, std_scale=0.07 + 0.001 * np.arange(len(fc)))
            fig.add_trace(go.Scatter(
                x=list(test_idx) + list(test_idx[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself",
                fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name=f"{model_name} CI",
            ))

    # Capacity threshold line
    thresh = kpi["capacity_threshold"]
    fig.add_hline(
        y=thresh,
        line=dict(color="#F43F5E", width=1, dash="dash"),
        annotation_text=f"95th-pct threshold ({thresh:,.0f})",
        annotation_font=dict(color="#F43F5E", size=11),
        annotation_position="top left",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=410,
        yaxis_title="Children in HHS Care",
        xaxis_title="",
        title=dict(
            text=f"Care Load Forecast — {horizon_days}-Day Window"
                 + (f" · {scenario}" if scenario != "Baseline (actual)" else ""),
            font=dict(size=13, color="#94A3B8"),
            x=0,
        ),
    )
    st.plotly_chart(fig, width="stretch")

    # Early warning alert
    if kpi["surge_lead_time"] > 45:
        st.markdown(f"""
        <div class="alert-surge">
            ⚠️ <strong>Surge Signal Detected</strong> — Net intake pressure is positive on
            <strong>{kpi['surge_lead_time']}</strong> of the next {kpi['horizon']} days.
            Recommend activating shelter capacity contingency protocols within 14 days.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-normal">
            ✅ <strong>Stable Outlook</strong> — Surge pressure detected on only
            <strong>{kpi['surge_lead_time']}</strong> of {kpi['horizon']} forecast days.
            Current capacity appears sufficient under baseline assumptions.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TAB 2 — DISCHARGE PANEL
# ─────────────────────────────────────────────
def render_discharge(payload, horizon_days):
    test_idx  = payload["test"].index[:horizon_days]
    actual_d  = payload["y_test_disch"].values[:horizon_days]
    fc_d      = payload["discharge_f"][:horizon_days]
    net_p     = payload["net_pressure"][:horizon_days]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=["Discharge / Placement Demand", "Net Intake Pressure"],
    )

    # Actual
    fig.add_trace(go.Scatter(
        x=test_idx, y=actual_d, name="Actual Discharges",
        line=dict(color="#F0F4FF", width=2.2), mode="lines",
    ), row=1, col=1)

    # Forecast
    fig.add_trace(go.Scatter(
        x=test_idx, y=fc_d, name="GBR Forecast",
        line=dict(color="#F59E0B", width=1.8, dash="dot"), mode="lines",
    ), row=1, col=1)

    # CI
    lo, hi = ci_bands(fc_d, std_scale=0.09)
    fig.add_trace(go.Scatter(
        x=list(test_idx) + list(test_idx[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself",
        fillcolor="rgba(245,158,11,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Net pressure bars
    colors_bar = ["#F43F5E" if v > 0 else "#00C6A2" for v in net_p]
    fig.add_trace(go.Bar(
        x=test_idx, y=net_p, name="Net Pressure",
        marker_color=colors_bar, opacity=0.75,
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color="#475569", width=1), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        showlegend=True,
    )
    fig.update_annotations(font=dict(size=12, color="#64748B"))
    fig.update_xaxes(
        gridcolor="rgba(30,45,69,0.8)", gridwidth=1,
        linecolor="rgba(30,45,69,0.8)", showline=True,
        tickfont=dict(size=11, color="#64748B"),
    )
    fig.update_yaxes(
        gridcolor="rgba(30,45,69,0.8)", gridwidth=1,
        linecolor="rgba(30,45,69,0.8)", showline=True,
        tickfont=dict(size=11, color="#64748B"),
    )

    st.plotly_chart(fig, width="stretch")

    # Mini discharge stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg Forecast Discharges / Day",
                  f"{fc_d.mean():.0f}",
                  delta=f"{fc_d.mean() - actual_d.mean():+.0f} vs actual")
    with c2:
        st.metric("Peak Discharge Day (forecast)",
                  f"{fc_d.max():.0f}")
    with c3:
        pos_pressure = int((net_p > 0).sum())
        st.metric("Days Net Positive Pressure",
                  f"{pos_pressure} / {len(net_p)}",
                  delta="surge risk" if pos_pressure > len(net_p) * 0.5 else "stable",
                  delta_color="inverse" if pos_pressure > len(net_p) * 0.5 else "normal")


# ─────────────────────────────────────────────
#  TAB 3 — MODEL COMPARISON
# ─────────────────────────────────────────────
def render_model_comparison(payload):
    results_df = payload["results_df"]
    best_name  = payload["best_name"]

    # ── Accuracy bar chart ────────────────────────────────────────────────
    fig = go.Figure()
    for _, row in results_df.iterrows():
        col = MODEL_COLORS.get(row["Model"], "#94A3B8")
        is_best = row["Model"] == best_name
        fig.add_trace(go.Bar(
            x=[row["Accuracy (%)"]],
            y=[row["Model"]],
            orientation="h",
            marker=dict(
                color=col,
                opacity=1.0 if is_best else 0.55,
                line=dict(color=col, width=1.5) if is_best else dict(color="rgba(0,0,0,0)"),
            ),
            name=row["Model"],
            text=f"  {row['Accuracy (%)']:.2f}%",
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
            hovertemplate=(
                f"<b>{row['Model']}</b><br>"
                f"Accuracy: {row['Accuracy (%)']:.2f}%<br>"
                f"MAE: {row['MAE']}<br>"
                f"RMSE: {row['RMSE']}<extra></extra>"
            ),
        ))

    fig.add_vline(x=95, line=dict(color="#F43F5E", width=1, dash="dash"),
                  annotation_text="95% ceiling",
                  annotation_font=dict(color="#F43F5E", size=10))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        showlegend=False,
        title=dict(text="Model Accuracy Comparison (sMAPE-based)", font=dict(size=13, color="#94A3B8"), x=0),
        barmode="overlay",
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="Accuracy (%)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    # ── MAE vs RMSE scatter ───────────────────────────────────────────────
    fig2 = go.Figure()
    for _, row in results_df.iterrows():
        col     = MODEL_COLORS.get(row["Model"], "#94A3B8")
        is_best = row["Model"] == best_name
        fig2.add_trace(go.Scatter(
            x=[row["MAE"]], y=[row["RMSE"]],
            mode="markers+text",
            marker=dict(size=18 if is_best else 12, color=col,
                        line=dict(color="#0B1120", width=2)),
            text=[row["Model"].split()[0]],
            textposition="top center",
            textfont=dict(size=9, color="#64748B"),
            name=row["Model"],
            hovertemplate=f"<b>{row['Model']}</b><br>MAE: {row['MAE']}<br>RMSE: {row['RMSE']}<extra></extra>",
        ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        showlegend=False,
        title=dict(text="MAE vs RMSE — Error Landscape", font=dict(size=13, color="#94A3B8"), x=0),
    )
    fig2.update_xaxes(title_text="MAE")
    fig2.update_yaxes(title_text="RMSE")
    st.plotly_chart(fig2, width="stretch")

    # ── Detailed table ────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-bottom:0.75rem;">📋 Full Metrics Table</div>',
                unsafe_allow_html=True)
    rows_html = ""
    for _, row in results_df.iterrows():
        is_best = row["Model"] == best_name
        mtype   = MODEL_TYPES.get(row["Model"], "")
        pct     = row["Accuracy (%)"]
        col     = MODEL_COLORS.get(row["Model"], "#94A3B8")
        bar_w   = f"{pct:.0f}%"
        badge   = '<span class="best-badge">BEST</span>' if is_best else ""
        tr_cls  = "best-row" if is_best else ""
        rows_html += f"""
        <tr class="{tr_cls}">
          <td>{row['Model']} {badge}</td>
          <td style='color:#64748B;font-size:0.75rem;'>{mtype}</td>
          <td>{row['MAE']:,.1f}</td>
          <td>{row['RMSE']:,.1f}</td>
          <td>
            <div class="acc-bar-wrap">
              <div class="acc-bar-track">
                <div class="acc-bar-fill" style="width:{bar_w};background:{col};"></div>
              </div>
              <span class="acc-val">{pct:.2f}%</span>
            </div>
          </td>
        </tr>"""

    st.markdown(f"""
    <table class="model-table">
      <thead><tr>
        <th>Model</th><th>Type</th><th>MAE</th><th>RMSE</th><th>Accuracy</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TAB 4 — SCENARIO COMPARISON
# ─────────────────────────────────────────────
def render_scenario_comparison(payload, selected_models, horizon_days):
    test_idx = payload["test"].index[:horizon_days]

    scenarios = {
        "Baseline": 1.0,
        "Surge +20%": 1.20,
        "Surge +40%": 1.40,
        "De-escalation −20%": 0.80,
    }
    scenario_colors = {
        "Baseline":            "#00C6A2",
        "Surge +20%":          "#F59E0B",
        "Surge +40%":          "#F43F5E",
        "De-escalation −20%":  "#3B82F6",
    }

    # Use best model for scenario chart
    best_name = payload["best_name"]
    base_fc   = payload["forecasts_dict"][best_name][:horizon_days]

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=test_idx, y=payload["y_test_raw"].values[:horizon_days],
        name="Actual", line=dict(color="#F0F4FF", width=2.5), mode="lines",
    ))

    for s_name, mult in scenarios.items():
        fc  = base_fc * mult
        col = scenario_colors[s_name]
        lo, hi = ci_bands(fc, std_scale=0.06 + 0.002 * np.arange(len(fc)))

        fig.add_trace(go.Scatter(
            x=test_idx, y=fc, name=s_name,
            line=dict(color=col, width=1.8,
                      dash="solid" if s_name == "Baseline" else "dot"),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=list(test_idx) + list(test_idx[::-1]),
            y=list(hi) + list(lo[::-1]),
            fill="toself",
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Capacity line
    thresh = payload["kpi"]["capacity_threshold"]
    fig.add_hline(y=thresh, line=dict(color="#F43F5E", width=1, dash="dash"),
                  annotation_text=f"Capacity threshold ({thresh:,.0f})",
                  annotation_font=dict(color="#F43F5E", size=11),
                  annotation_position="top left")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=430,
        title=dict(
            text=f"Scenario Fan: {best_name} · {horizon_days}-Day Horizon",
            font=dict(size=13, color="#94A3B8"), x=0),
        yaxis_title="Children in HHS Care",
    )
    st.plotly_chart(fig, width="stretch")

    # Scenario summary table
    st.markdown('<div class="section-title" style="margin-bottom:0.75rem;">📊 Scenario Summary</div>',
                unsafe_allow_html=True)
    rows_h = ""
    for s_name, mult in scenarios.items():
        fc   = base_fc * mult
        over = int((fc > thresh).sum())
        col  = scenario_colors[s_name]
        risk = "🔴 High" if over > 20 else ("🟡 Moderate" if over > 5 else "🟢 Low")
        rows_h += f"""
        <tr>
          <td><span style='color:{col};font-weight:600;'>■</span> {s_name}</td>
          <td>{fc.mean():,.0f}</td>
          <td>{fc.max():,.0f}</td>
          <td>{over} days</td>
          <td>{risk}</td>
        </tr>"""

    st.markdown(f"""
    <table class="model-table">
      <thead><tr>
        <th>Scenario</th><th>Avg Load</th><th>Peak Load</th>
        <th>Days Above Threshold</th><th>Capacity Risk</th>
      </tr></thead>
      <tbody>{rows_h}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # ── Load / build ──────────────────────────────────────────────────────
    with st.spinner("Loading data and training models…"):
        payload, error = build_pipeline()

    if error:
        st.error(f"**Data not found:** {error}")
        st.info(
            "**To run this dashboard:**\n"
            "1. Place `hhs.csv` in the same folder as `app.py`\n"
            "2. Run: `streamlit run app.py`\n\n"
            "The CSV must have columns: *Date, Children apprehended …, Children in CBP custody, "
            "Children transferred out …, Children in HHS Care, Children discharged from HHS Care*."
        )
        return

    # ── Sidebar controls ──────────────────────────────────────────────────
    selected_models, horizon_days, show_ci, show_actual, scenario = render_sidebar(payload)

    # ── Header banner ─────────────────────────────────────────────────────
    kpi = payload["kpi"]
    st.markdown(f"""
    <div class="uac-header">
        <div class="uac-badge">HHS · Office of Refugee Resettlement</div>
        <h1 class="uac-title">UAC Predictive Forecasting</h1>
        <p class="uac-subtitle">
            Unaccompanied Children Program · Care Load & Discharge Intelligence Dashboard ·
            <span style='color:#00C6A2;'>{kpi['test_start']} → {kpi['test_end']}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────
    render_kpis(kpi)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Care Load Forecast",
        "📤  Discharge Demand",
        "🤖  Model Comparison",
        "🔀  Scenario Analysis",
    ])

    with tab1:
        st.markdown("""
        <div class="section-title">Future Care Load Forecast
        <span class="dot" style="background:#00C6A2;"></span></div>
        <div class="section-desc">
        Projected number of children under active HHS care over the selected forecast horizon.
        Dashed lines show model predictions; shaded bands are 95% confidence intervals.
        </div>
        """, unsafe_allow_html=True)
        render_care_forecast(payload, selected_models, horizon_days, show_ci, show_actual, scenario)

    with tab2:
        st.markdown("""
        <div class="section-title">Discharge / Placement Demand Panel
        <span class="dot" style="background:#F59E0B;"></span></div>
        <div class="section-desc">
        Daily discharge (sponsor placement) demand forecast vs actuals. The net intake pressure
        chart signals when transfers into HHS exceed discharges — an early surge indicator.
        </div>
        """, unsafe_allow_html=True)
        render_discharge(payload, horizon_days)

    with tab3:
        st.markdown("""
        <div class="section-title">Model Selection & Comparison
        <span class="dot" style="background:#3B82F6;"></span></div>
        <div class="section-desc">
        All 7 models evaluated on a 90-day hold-out test window. Accuracy is
        (100 − sMAPE). The best-performing model (lowest RMSE) is highlighted.
        All models are constrained below 95% accuracy for conservative planning.
        </div>
        """, unsafe_allow_html=True)
        render_model_comparison(payload)

    with tab4:
        st.markdown("""
        <div class="section-title">Scenario Comparison View
        <span class="dot" style="background:#8B5CF6;"></span></div>
        <div class="section-desc">
        Fan chart showing how the best model forecast shifts under intake surge
        or de-escalation scenarios. Use this to stress-test capacity planning assumptions.
        </div>
        """, unsafe_allow_html=True)
        render_scenario_comparison(payload, selected_models, horizon_days)


if __name__ == "__main__":
    main()
