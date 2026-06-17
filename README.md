# 🏛️ UAC Predictive Forecasting Dashboard

**A time-series forecasting and capacity-planning tool for the HHS Office of Refugee Resettlement's Unaccompanied Children (UAC) program**, built with Python, scikit-learn, statsmodels, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-TimeSeries-3C5A99)](https://www.statsmodels.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

[Live Demo](#) · [Notebook Walkthrough](#notebook) · [Report a Bug](#)

---

## Overview

The U.S. Department of Health and Human Services (HHS) Office of Refugee Resettlement is responsible for the care and placement of unaccompanied children (UAC) who arrive at the U.S. border. Care loads can shift quickly, and shelters, staffing, and sponsor-placement pipelines all depend on being able to see a surge coming before it happens.

This project turns publicly structured HHS/CBP intake and care-load data into an **interactive forecasting dashboard** that:

- Projects how many children will be in active HHS care over the next 7–90 days
- Forecasts daily discharge (sponsor placement) demand
- Flags the probability of breaching shelter capacity thresholds
- Lets planners stress-test "what if intake surges 40%?" scenarios before they happen

It was built end-to-end — from raw data cleaning through feature engineering, model benchmarking, and a production-style Streamlit front end.

## Demo

>  https://predictive-forecasting-of-care-load-placement-demand-0.streamlit.app


## Key Features

- **Multi-model forecasting engine** — seven models spanning naive baselines, classical statistical methods, and machine learning, benchmarked side by side on a held-out test window
- **Four-tab analyst workflow** — Care Load Forecast, Discharge Demand, Model Comparison, and Scenario Analysis
- **Scenario stress-testing** — simulate intake surges (+20%, +40%) or de-escalations (−20%) and see how the forecast and capacity risk shift in real time
- **Capacity early-warning system** — KPI strip surfaces forecast accuracy, surge lead time, capacity breach probability, forecast stability, and discharge model accuracy at a glance
- **Confidence interval bands** on every forecast trace, so uncertainty is visible, not hidden
- **Net intake pressure tracking** — a derived signal (transfers in vs. discharges out) that gives an early read on building system pressure
- **Self-contained pipeline** — no pre-trained model files required; data is cleaned, features are engineered, and all models are trained on demand, then cached for performance

## How It Works

**1. Data cleaning** — Raw HHS daily records (children apprehended, in CBP custody, transferred, in HHS care, discharged) are parsed, de-duplicated, reindexed to a continuous daily calendar, and gap-filled.

**2. Feature engineering** — Lag features (7/14/30/60 days) and rolling mean/std windows are generated for care load, discharges, and a derived "net pressure" signal, alongside calendar features (day of week, month, quarter, weekend flag).

**3. Train/test split** — The most recent 90 days are held out as a test window to simulate real forecasting conditions.

**4. Model benchmarking** — Seven models are trained and evaluated head-to-head:

| Category | Models |
|---|---|
| Baseline | Naive Persistence, 7-Day Moving Average |
| Statistical | ARIMA(0,1,0), Exponential Smoothing |
| Machine Learning | Random Forest, Gradient Boosting, Ridge Regression |

Each is scored on **MAE**, **RMSE**, and an **sMAPE-based accuracy**, and the lowest-RMSE model is automatically promoted to "best model" for the live forecast view. A dedicated Gradient Boosting model handles discharge-demand forecasting separately.

**5. Capacity intelligence** — A 95th-percentile capacity threshold is computed from historical care-load data, and the dashboard estimates the probability that the forecast breaches it — turning a raw forecast into an actionable planning signal.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| Forecasting / ML | scikit-learn (Random Forest, Gradient Boosting, Ridge), statsmodels (ARIMA, Exponential Smoothing) |
| Visualization | Plotly |
| App / UI | Streamlit, custom CSS theming |
| Persistence | joblib |

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit numpy pandas plotly scikit-learn statsmodels joblib
```

> Prefer a single command? Generate a `requirements.txt` from the imports above and run `pip install -r requirements.txt`.

### Running the app

1. Place the source data file `hhs.csv` in the project root (same folder as `app.py`). The CSV should contain the following columns: `Date`, `Children apprehended and placed in CBP custody`, `Children in CBP custody`, `Children transferred out of CBP custody`, `Children in HHS Care`, `Children discharged from HHS Care`.
2. Launch the dashboard:

```bash
streamlit run app.py
```

3. Open the local URL Streamlit prints in your terminal (typically `http://localhost:8501`).

## Project Structure

```
.
├── app.py                                   # Streamlit dashboard (data pipeline, models, UI)
├── UAC_Predictive_Forecasting_v2_fixed.ipynb # Exploratory analysis & model development notebook
├── hhs.csv                                   # Source data (not included — add your own)
└── README.md
```

## Notebook

The accompanying Jupyter notebook (`UAC_Predictive_Forecasting_v2_fixed.ipynb`) documents the full analytical process behind the dashboard, step by step: data loading and inspection, cleaning and preprocessing, exploratory data analysis, feature engineering, the train/test split, baseline and statistical modeling, machine learning model development, evaluation, forecast visualization, discharge-demand modeling, KPI derivation, and artifact preparation for deployment. It's a good companion read if you want to see the reasoning behind each modeling decision before diving into the dashboard code.

## Roadmap

- [ ] Swap in live HHS/CBP data feeds instead of a static CSV
- [ ] Add SARIMA / Prophet / LSTM as additional candidate models
- [ ] Hyperparameter tuning with time-series cross-validation
- [ ] Persist trained models with `joblib` to skip retraining on each app restart
- [ ] Deploy to Streamlit Community Cloud and link the live demo above
- [ ] Add automated tests for the data pipeline and evaluation functions

## Author

**Sarfaraz Ali**


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<sub>Built as part of the Unified Mentor program. This project uses a publicly structured dataset format and is intended for educational and portfolio purposes.</sub>
