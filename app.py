# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timezone, timedelta

# Optional: astral for real sunrise/sunset
try:
    from astral import LocationInfo
    from astral.sun import sun
    ASTRAL_OK = True
except Exception:
    ASTRAL_OK = False

# UI CSS (dark glossy / label visibility)
st.set_page_config(page_title="Smart Weather Predictor", layout="wide", page_icon="🌦️")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top left, #0f2b33 0%, #12333a 30%, #0e2a30 100%);
  color: #eaf6ff;
  min-height: 100vh;
  padding-top: 30px;
}
h1 { color: #00d0ff !important; text-shadow: 0 4px 18px rgba(0,208,255,0.25); }
label, .stMarkdown, .css-18e3th9, .css-hxt7ib {
  color: #eaf6ff !important;
  font-weight: 600;
}
div[data-baseweb="input"] input, .stNumberInput input {
  color: #012 !important;
  background: rgba(255,255,255,0.92) !important;
  border-radius: 10px !important;
  padding-left: 14px !important;
}
div.stButton > button {
  color: white !important;
  background: linear-gradient(90deg,#007bff,#00d4ff) !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  box-shadow: 0 6px 20px rgba(0,208,255,0.15);
}
.stAlert {
  border-radius: 12px !important;
  color: #ffffff !important;
  font-weight: 800 !important;
}
div[data-testid="stAlert"][class*="st-success"] * {
  color: #baffc9 !important;
  text-shadow: 0 0 6px rgba(186,255,201,0.5);
  font-weight: 700 !important;
}
div[data-testid="stAlert"][class*="st-success"] {
  background-color: #003c00 !important;
  border: 1px solid #00ff6a !important;
}
div[data-testid="stAlert"][class*="st-info"] * {
  color: #a8d8ff !important;
  text-shadow: 0 0 6px rgba(168,216,255,0.4);
  font-weight: 700 !important;
}
div[data-testid="stAlert"][class*="st-info"] {
  background-color: #002a66 !important;
  border: 1px solid #1e90ff !important;
}
.css-1v3fvcr { color: #dbefff !important; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌦️ Smart Weather Predictor</h1>", unsafe_allow_html=True)
st.markdown("AI + heuristic rain probability. Enter current local conditions and tap Predict.")

# try load model & feature order
MODEL_FN = "weather_model.zst"
FEATURES_FN = "weather_features.pkl"
model = None
feature_names = None
try:
    model = joblib.load(MODEL_FN)
    st.info("✅ Model loaded.")
except Exception as e:
    st.warning(f"Model not found or failed to load ({MODEL_FN}). Falling back to heuristic. ({e})")

try:
    feature_names = joblib.load(FEATURES_FN)
except Exception:
    feature_names = None

# Input fields
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.5, min_value=-50.0, max_value=60.0)
    humidity = st.number_input("💧 Humidity (%)", value=70.0, min_value=0.0, max_value=100.0)
    pressure = st.number_input("📉 Pressure (hPa)", value=1010.0, min_value=800.0, max_value=1100.0)
with col2:
    wind_speed = st.number_input("💨 Wind Speed (km/h)", value=5.0, min_value=0.0, max_value=200.0)
    visibility = st.number_input("👀 Visibility (km)", value=10.0, min_value=0.0, max_value=100.0)
    wind_direction = st.number_input("🧭 Wind Direction (°)", value=45.0, min_value=0.0, max_value=360.0)

# --- City Selection ---
city_options = {
    "Bhubaneswar": (20.2961, 85.8245),
    "Ranchi": (23.3441, 85.3096),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639)
}
selected_city = st.selectbox("Choose city:", list(city_options.keys()))
lat, lon = city_options[selected_city]

# helper: compute sunset
from suntime import Sun
def get_sun_times(lat, lon):
    sun = Sun(lat, lon)
    today = datetime.now().date()
    today_dt = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
    sunrise_utc = sun.get_sunrise_time(today_dt)
    sunset_utc = sun.get_sunset_time(today_dt)
    local_offset = timedelta(hours=5, minutes=30)
    sunrise_local = sunrise_utc + local_offset
    sunset_local = sunset_utc + local_offset
    day_length = sunset_local - sunrise_local
    return sunrise_local.strftime("%H:%M:%S"), sunset_local.strftime("%H:%M:%S"), str(day_length).split(".")[0]

# helper: build input dataframe
def build_input_df(feature_names):
    if feature_names:
        vals = {}
        for f in feature_names:
            if "Temperature" in f:
                vals[f] = temperature
            elif "Humidity" in f:
                vals[f] = humidity
            elif "Pressure" in f or "millibar" in f.lower():
                vals[f] = pressure
            elif "Wind Speed" in f:
                vals[f] = wind_speed
            elif "Visibility" in f:
                vals[f] = visibility
            elif "Wind Bearing" in f or "Wind Direction" in f or "wind" in f.lower() and "bearing" in f.lower():
                vals[f] = wind_direction
            elif f in ("year", "month", "day", "hour"):
                now = datetime.now()
                vals[f] = getattr(now, f)
            else:
                vals[f] = 0.0
        return pd.DataFrame([vals], columns=feature_names)
    return pd.DataFrame([{
        "Temperature (C)": temperature,
        "Humidity": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Wind Bearing (degrees)": wind_direction,
        "Visibility (km)": visibility,
        "Loud Cover": 0.0,
        "Pressure (millibars)": pressure
    }])

# core: compute rain chance
def compute_rain_chance_with_model(df):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            if probs.shape[1] >= 2:
                p = probs[0, 1]
                return float(np.clip(p * 100, 0, 100))
        pred = model.predict(df)
        p0 = float(np.asarray(pred).ravel()[0])
        if 0.0 <= p0 <= 1.0:
            return float(np.clip(p0 * 100, 0, 100))
    except Exception:
        pass
    return None

def heuristic_rain(humidity, visibility, pressure, wind_speed, lat=20.2961, lon=85.8245):
    base = humidity * 0.65
    vis_penalty = max(0, 12 - visibility) * 3.5
    pressure_factor = max(0, 1015 - pressure) * 0.5
    wind_factor = min(20, wind_speed) * 0.6
    coastal_bonus = 5 if abs(lat) < 25 else 0
    inland_penalty = -3 if abs(lat) > 28 else 0
    raw = base + vis_penalty + pressure_factor + wind_factor + coastal_bonus + inland_penalty
    return float(np.clip(raw, 0, 100))

# Predict action
if st.button("🔮 Predict Rain Chance"):
    input_df = build_input_df(feature_names)
    rain_chance = None
    used_model = False

    if model is not None:
        rain_chance = compute_rain_chance_with_model(input_df)
        if rain_chance is not None:
            used_model = True

    if rain_chance is None:
        rain_chance = heuristic_rain(humidity, visibility, pressure, wind_speed, lat, lon)

    st.success(f"**Predicted Rain Chance: {rain_chance:.2f}%**")

    if rain_chance >= 70:
        st.warning("⚠️ High chance of rain — carry an umbrella!")
        st.markdown("🟦 **Tip:** Keep a raincoat/umbrella handy.")
    elif rain_chance >= 40:
        st.info("🌦 Moderate chance of rain — light rain gear advised.")
        st.markdown("🟡 Prediction provided for guidance only — consult official forecasts or radar for confirmation.")
        st.caption("📝 The predicted rain chance is calculated according to the parameters entered by the user.")
        st.write("")
        st.markdown("<p style='text-align:center; font-size:13px; color:gray;'>Developed and built by Aditya Kumar</p>", unsafe_allow_html=True)


    else:
        st.success("☀️ Low chance of rain — enjoy your day!")

    sunrise_time, sunset_time, day_length = get_sun_times(lat, lon)
    st.markdown(f"🌅 **Sunrise Time in {selected_city}:** {sunrise_time}")
    st.markdown(f"🏙️ **Sunset Time in {selected_city}:** {sunset_time}")
    st.markdown(f"🕰️ **Day Length:** {day_length} hours")
    st.caption(f"🕒 Prediction made at {datetime.now().strftime('%I:%M %p, %d %B %Y')}")

    if used_model:
        st.caption("Prediction produced by the loaded model.")
    else:
        st.caption("Prediction produced by a local heuristic (model not available or not probability).")

if st.checkbox("Show input features used (debug)"):
    st.dataframe(build_input_df(feature_names).T)

st.markdown("---")
st.markdown("Built for interactive demos — improves with real model & correct feature order.")



