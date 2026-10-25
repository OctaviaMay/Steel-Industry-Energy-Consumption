
import os
from datetime import datetime, time

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Steel Plant Energy Forecast", page_icon="⚡", layout="wide")

MODELS = {"Random Forest (recommended)": "random_forest", "XGBoost": "xgboost"}
LOAD_TYPES = ["Light_Load", "Medium_Load", "Maximum_Load"]
TIMEOUT = 60 


def get_api_url():
    try:
        url = st.secrets["API_URL"]
    except (KeyError, FileNotFoundError):
        url = os.getenv("API_URL", "http://127.0.0.1:8000")
    return url.rstrip("/")


API_URL = get_api_url()


# --------------------------------------------------------------------------
# API calls
# --------------------------------------------------------------------------

class ApiError(Exception):
    pass


def call_api(method, path, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Can't reach the API at {API_URL}. Is it running? ({type(exc).__name__})")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"API returned {resp.status_code}: {detail}")
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_summary(model):
    return call_api("GET", "/evaluation/summary", params={"model": model})


@st.cache_data(ttl=3600, show_spinner=False)
def get_series(model, start, days):
    return call_api("GET", "/evaluation/series",
                    params={"model": model, "start": start.isoformat(), "days": days})


def post_predict(payload):
    return call_api("POST", "/predict", json=payload)


def mae_bar_chart(mae_dict, order=None):
    data = pd.DataFrame({"group": list(mae_dict), "MAE": list(mae_dict.values())})
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("group:N", sort=order, title=None),
        y=alt.Y("MAE:Q", title="MAE (kWh)"),
        tooltip=["group", alt.Tooltip("MAE:Q", format=".2f")],
    ).properties(height=260)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Settings")
model_label = st.sidebar.radio("Model", list(MODELS))
model = MODELS[model_label]

try:
    with st.spinner("Connecting to the forecasting API..."):
        summary = get_summary(model)
except ApiError as err:
    st.error(str(err))
    st.stop()

first_day = pd.Timestamp(summary["test_period_start"]).date()
last_day = pd.Timestamp(summary["test_period_end"]).date()
start = st.sidebar.date_input("Start date", value=pd.Timestamp("2018-11-12").date(),
                              min_value=first_day, max_value=last_day)
days = st.sidebar.slider("Days to show", 1, 14, 3)
st.sidebar.caption(
    f"Test period: {first_day:%d %b} – {last_day:%d %b %Y}. "
    "The model never saw this data during training."
)
st.sidebar.caption(f"API: `{API_URL}` · [docs]({API_URL}/docs)")

# --------------------------------------------------------------------------
# Header and KPIs
# --------------------------------------------------------------------------
st.title("⚡ Steel Plant Electricity Forecast")
st.markdown(
    "Predicts the plant's electricity use **15 minutes ahead**, using only information "
    "available *before* that period starts. Data: 2018 readings from a steel plant in "
    "South Korea ([UCI dataset #851](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption))."
)

overall = summary["overall"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² (full test period)", f"{overall['r2']:.3f}")
c2.metric("Model error (MAE)", f"{overall['mae_kwh']:.2f} kWh")
c3.metric("Naive baseline error", f"{overall['naive_mae_kwh']:.2f} kWh",
          help="Baseline: predict that the next 15 minutes equal the last 15 minutes.")
c4.metric("Better than baseline by", f"{overall['improvement_over_naive']:.0%}")

tab_history, tab_predict = st.tabs(["📈 Test period results", "🔮 Try a prediction"])

# --------------------------------------------------------------------------
# Tab 1: actual vs predicted, error breakdown
# --------------------------------------------------------------------------
with tab_history:
    st.subheader("Actual vs predicted")
    try:
        series = get_series(model, datetime.combine(start, time()), days)
    except ApiError as err:
        st.info(str(err))
        series = None

    if series:
        show_naive = st.checkbox("Show naive baseline", value=False)
        points = pd.DataFrame(series["points"])
        points["timestamp"] = pd.to_datetime(points["timestamp"])
        names = {"actual_kwh": "Actual", "predicted_kwh": "Predicted", "naive_kwh": "Naive baseline"}
        cols = ["actual_kwh", "predicted_kwh"] + (["naive_kwh"] if show_naive else [])
        long = points.melt(id_vars="timestamp", value_vars=cols, var_name="series", value_name="kWh")
        long["series"] = long["series"].map(names)

        chart = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("kWh:Q", title="Usage (kWh per 15 min)"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
            strokeDash=alt.condition(alt.datum.series == "Naive baseline",
                                     alt.value([4, 3]), alt.value([1, 0])),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("kWh:Q", format=".2f")],
        ).properties(height=380).interactive(bind_y=False)
        st.altair_chart(chart, width="stretch")

        m = series["metrics"]
        st.caption(
            f"Selected {days} day(s): MAE {m['mae_kwh']:.2f} kWh vs baseline "
            f"{m['naive_mae_kwh']:.2f} kWh ({m['improvement_over_naive']:.0%} better)."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Error by load type")
        st.altair_chart(mae_bar_chart(summary["mae_by_load_type"], LOAD_TYPES), width="stretch")
        st.caption("Heavier load periods are harder to forecast: consumption swings more.")
    with right:
        st.subheader("Error by day type")
        by_day = summary["mae_by_day_type"]
        st.altair_chart(mae_bar_chart(by_day), width="stretch")
        gap = max(by_day.values()) - min(by_day.values())
        st.caption(f"Weekday/weekend gap: {gap:.2f} kWh (KPI target: under 5 kWh).")

# --------------------------------------------------------------------------
# Tab 2: single prediction through POST /predict
# --------------------------------------------------------------------------
with tab_predict:
    st.markdown("Enter the plant's latest readings to forecast the **next 15 minutes**.")

    with st.form("predict_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            p_date = st.date_input("Date", value=pd.Timestamp("2018-11-12").date())
            p_time = st.time_input("Time of the reading to forecast", value=time(10, 0), step=900)
            p_load = st.selectbox("Load type", LOAD_TYPES, index=1)
        with col_b:
            p_reactive = st.number_input("Lagging reactive power (kVarh)", 0.0, 200.0, 32.4)
            p_pf = st.number_input("Lagging power factor (%)", 0.0, 100.0, 78.5)

        st.markdown("**Previous four readings (kWh), oldest first**")
        usage_cols = st.columns(4)
        defaults = [52.1, 49.8, 55.3, 51.0]
        recent = [
            usage_cols[i].number_input(f"{60 - 15 * i} min ago",
                                       0.0, 200.0, defaults[i], key=f"usage_{i}")
            for i in range(4)
        ]
        submitted = st.form_submit_button("Forecast", type="primary")

    if submitted:
        payload = {
            "timestamp": datetime.combine(p_date, p_time).isoformat(),
            "load_type": p_load,
            "recent_usage_kwh": recent,
            "lagging_reactive_power_kvarh": p_reactive,
            "lagging_power_factor": p_pf,
            "model": model,
        }
        try:
            result = post_predict(payload)
        except ApiError as err:
            st.error(str(err))
        else:
            r1, r2 = st.columns(2)
            r1.metric("Forecast for the next 15 minutes", f"{result['predicted_usage_kwh']:.1f} kWh")
            r2.metric("Naive baseline (last reading)", f"{result['naive_baseline_kwh']:.1f} kWh")
            with st.expander("Features the API built from the inputs"):
                st.json(result["features"])

# --------------------------------------------------------------------------
with st.expander("How it works"):
    st.markdown(
        f"""
- **Leakage-aware features.** CO₂ was dropped (r = 0.988 with usage, only 8 unique values:
  derived from usage, not measured independently).
- **Chronological split.** Trained on the first 70% of the year, tested on the last 30%,
  so the test set is a genuinely future period.
- **Business KPIs, not just accuracy:** R², improvement over a naive baseline,
  error by load type, and weekday/weekend stability.

Code: [GitHub](https://github.com/OctaviaMay/Steel-Industry-Energy-Consumption)
"""
    )