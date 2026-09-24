

import streamlit as st
import numpy as np
import joblib
import json
import math

from tensorflow.keras.models import load_model

# Page Config 
st.set_page_config(
    page_title=" Bus Passenger Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1565C0; text-align: center;
        padding: 10px 0; border-bottom: 3px solid #1565C0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        border-radius: 12px; padding: 20px;
        color: white; text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 2.5rem; font-weight: 800; }
    .metric-label { font-size: 0.95rem; opacity: 0.85; margin-top: 4px; }
    .badge-low    { background:#4CAF50; color:white; padding:4px 12px; border-radius:20px; }
    .badge-medium { background:#FF9800; color:white; padding:4px 12px; border-radius:20px; }
    .badge-high   { background:#F44336; color:white; padding:4px 12px; border-radius:20px; }
    .interval-row {
        display:flex; justify-content:space-between; align-items:center;
        background:#F0F4FF; border-radius:8px; padding:10px 16px; margin:6px 0;
    }
    .stButton>button {
        width:100%; background:linear-gradient(90deg,#1565C0,#1976D2);
        color:white; font-size:1.1rem; font-weight:600;
        border:none; border-radius:10px; padding:14px;
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Artifacts 
@st.cache_resource
def load_artifacts():
    model     = load_model('best_bilstm_model.h5')
    scaler_X  = joblib.load('scaler_X.pkl')
    scaler_y  = joblib.load('scaler_y.pkl')
    le_tod    = joblib.load('le_tod.pkl')
    le_dow    = joblib.load('le_dow.pkl')
    le_stop   = joblib.load('le_stop.pkl')
    le_dest   = joblib.load('le_dest.pkl')
    with open('model_config.json') as f:
        config = json.load(f)
    return model, scaler_X, scaler_y, le_tod, le_dow, le_stop, le_dest, config

model, scaler_X, scaler_y, le_tod, le_dow, le_stop, le_dest, config = load_artifacts()

SEQ_LEN      = config['seq_len']
BUS_CAPACITY = config['bus_capacity']
FEATURE_COLS = config['feature_cols']

# Helpers
def parse_time_slot(t):
    start = t.split('-')[0]
    h, m  = start.split(':')
    return int(h) + int(m) / 60

def bin_demand(count):
    if count <= 50:   return " Low",    "badge-low"
    elif count <= 100: return " Medium", "badge-medium"
    else:              return " High",   "badge-high"

def build_sequence(row_features, scaler_X, seq_len):
    """Replicate single row into a sequence (suitable for real-time single input)."""
    row_scaled = scaler_X.transform([row_features])
    sequence   = np.tile(row_scaled, (seq_len, 1))
    return sequence.reshape(1, seq_len, len(row_features))

def forecast_intervals(model, seq, scaler_y, steps=4):
    preds = []
    s = seq.copy()
    for _ in range(steps):
        p = model.predict(s, verbose=0)
        val = max(scaler_y.inverse_transform(p)[0][0], 0)
        preds.append(val)
        new_row = s[0, -1, :].copy()
        new_row[9] = new_row[8]    
        new_row[8] = p[0][0]        
        s = np.concatenate([s[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)
    return preds

#  Generate all 30-min time slots
TIME_SLOTS = []
for h in range(5, 22):
    for m in [0, 30]:
        start = f"{h:02d}:{m:02d}"
        end_h, end_m = (h, m + 30) if m == 0 else (h + 1, 0)
        end   = f"{end_h:02d}:{end_m:02d}"
        TIME_SLOTS.append(f"{start}-{end}")

#  UI
st.markdown('<div class="main-header"> Bus Passenger Count Predictor</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3], gap="large")

# Input Panel 
with col_left:
    st.markdown("### Enter Route Details")

    time_slot  = st.selectbox(" Time Slot", TIME_SLOTS, index=4)
    day_of_week = st.selectbox(" Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    col_a, col_b = st.columns(2)
    with col_a:
        time_of_day = st.selectbox(" Time of Day",
            sorted(config['time_of_day_classes']))
    with col_b:
        peak_status = st.selectbox(" Peak / Off-Peak",
            ["Peak", "Off-Peak"])

    col_c, col_d = st.columns(2)
    with col_c:
        bus_stop = st.selectbox(" Bus Stop",
            sorted(config['busstop_classes']))
    with col_d:
        destination = st.selectbox(" Destination",
            sorted(config['destination_classes']))

    col_e, col_f = st.columns(2)
    with col_e:
        holiday = st.selectbox(" Holiday / Event", ["No", "Yes"])
    with col_f:
        bus_capacity = st.number_input(" Bus Capacity",
            min_value=20, max_value=200, value=70, step=5)

    st.markdown("---")
    st.markdown("** Historical Lag Values** *(last 2 observed counts)*")
    col_g, col_h = st.columns(2)
    with col_g:
        lag1 = st.number_input("Lag t-1 (previous slot)", min_value=0, max_value=300, value=75)
    with col_h:
        lag2 = st.number_input("Lag t-2 (2 slots ago)",   min_value=0, max_value=300, value=68)

    predict_btn = st.button(" Predict Now", use_container_width=True)

# Output Panel 
with col_right:
    st.markdown("###  Prediction Results")

    if predict_btn:
        #Encode inputs
        try:
            tod_enc  = le_tod.transform([time_of_day])[0]
        except:
            tod_enc  = 0
        try:
            dow_enc  = le_dow.transform([day_of_week])[0]
        except:
            dow_enc  = 0
        try:
            stop_enc = le_stop.transform([bus_stop])[0]
        except:
            stop_enc = 0
        try:
            dest_enc = le_dest.transform([destination])[0]
        except:
            dest_enc = 0

        hour_num  = parse_time_slot(time_slot)
        peak_enc  = 1 if peak_status == "Peak" else 0
        hol_enc   = 1 if holiday == "Yes" else 0

        row_features = [
            hour_num, tod_enc, dow_enc,
            peak_enc, hol_enc, bus_capacity,
            stop_enc, dest_enc,
            float(lag1), float(lag2)
        ]

        # Build sequence & predict
        seq = build_sequence(row_features, scaler_X, SEQ_LEN)
        pred_scaled = model.predict(seq, verbose=0)
        pred_passengers = max(scaler_y.inverse_transform(pred_scaled)[0][0], 0)
        pred_buses      = math.ceil(pred_passengers / bus_capacity)
        demand_label, demand_css = bin_demand(pred_passengers)

        # Main metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(pred_passengers)}</div>
                <div class="metric-label">🧍 Predicted Passengers</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="background:linear-gradient(135deg,#2E7D32,#1B5E20)">
                <div class="metric-value">{pred_buses}</div>
                <div class="metric-label"> Buses Required</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        #Demand badge
        st.markdown(f"**Demand Level:** &nbsp;<span class='{demand_css}'>{demand_label}</span>",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("** Summary**")
        summary = {
            "Route":           f"{bus_stop} → {destination}",
            "Time Slot":       time_slot,
            "Day":             day_of_week,
            "Peak Status":     peak_status,
            "Holiday":         holiday,
            "Pred. Passengers": int(pred_passengers),
            "Buses Needed":    pred_buses,
            "Capacity Left":   max(pred_buses * bus_capacity - int(pred_passengers), 0)
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame(list(summary.items()), columns=["Field", "Value"]),
            hide_index=True, use_container_width=True
        )

    else:
        st.info(" Fill in the route details on the left, then click **Predict Now**.")
        st.markdown("""
        

                    
        """)

# Footer 
st.markdown("---")
st.markdown(
    "<center><small>BiLSTM Passenger Prediction Model · "
    "Built with TensorFlow + Streamlit</small></center>",
    unsafe_allow_html=True
)
