

import os
import math
import json
import warnings
import numpy as np
import joblib
from datetime import datetime, timezone
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify
from flask_cors import CORS
from users import users_bp
from predictions import predict_bp
from flask_jwt_extended import (
    JWTManager, jwt_required, get_jwt_identity
)
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

#  Load .env first 
load_dotenv()

#  Import DB and models 
from database import db, init_db
from models import User, Prediction
from auth import auth_bp   # Import the auth blueprint



# APP SETUP

app = Flask(__name__)
@app.after_request
def handle_options(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": "http://localhost:5173"}}
) 
 # Allow all origins — restrict specific origins in production

#  JWT configuration 
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "change-this-secret")
jwt = JWTManager(app)

#  Register auth routes (/auth/register, /auth/login, /auth/me) 
app.register_blueprint(auth_bp)

app.register_blueprint(users_bp)        
app.register_blueprint(predict_bp) 

# Connect PostgreSQL database 
init_db(app)


# LOAD ML MODEL & ARTIFACTS (once at startup)
print("Loading model and artifacts.")
model    = load_model("best_bilstm_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
le_tod   = joblib.load("le_tod.pkl")
le_dow   = joblib.load("le_dow.pkl")
le_stop  = joblib.load("le_stop.pkl")
le_dest  = joblib.load("le_dest.pkl")

with open("model_config.json") as f:
    config = json.load(f)

SEQ_LEN      = config["seq_len"]
BUS_CAPACITY = config["bus_capacity"]
print(" Model ready!")



# HELPER FUNCTIONS


def parse_time_slot(t):
    """Convert '07:00-07:30' → 7.0 (decimal hour)."""
    start = t.split("-")[0]
    h, m  = start.split(":")
    return int(h) + int(m) / 60


def get_hour_from_slot(t):
    """Return integer hour from time slot string (for DB filtering)."""
    return int(t.split("-")[0].split(":")[0])


def bin_demand(count):
    """Map passenger count → Low / Medium / High."""
    if count <= 50:    return "Low"
    elif count <= 100: return "Medium"
    else:              return "High"


def build_sequence(row_features, seq_len):
    """Scale features and tile into LSTM sequence shape."""
    row_scaled = scaler_X.transform([row_features])
    sequence   = np.tile(row_scaled, (seq_len, 1))
    return sequence.reshape(1, seq_len, len(row_features))


def run_prediction(data):
    """
    Core prediction logic shared by /predict and /forecast.
    Takes the parsed JSON body, returns (row_features, seq, bus_cap).
    """
    hour_num  = parse_time_slot(data["time_slot"])
    tod_enc   = int(le_tod.transform([data["time_of_day"]])[0])
    dow_enc   = int(le_dow.transform([data["day_of_week"]])[0])
    stop_enc  = int(le_stop.transform([data["bus_stop"]])[0])
    dest_enc  = int(le_dest.transform([data["destination"]])[0])
    peak_enc  = 1 if data["peak_status"] == "Peak" else 0
    hol_enc   = 1 if data["holiday"] == "Yes" else 0
    bus_cap   = int(data.get("bus_capacity", BUS_CAPACITY))
    lag1      = float(data["lag_t1"])
    lag2      = float(data["lag_t2"])

    row_features = [
        hour_num, tod_enc, dow_enc,
        peak_enc, hol_enc, bus_cap,
        stop_enc, dest_enc,
        lag1, lag2
    ]
    seq = build_sequence(row_features, SEQ_LEN)
    return row_features, seq, bus_cap


def forecast_intervals(seq, bus_cap, steps=4):
    """Auto-regressive multi-step forecast (+15, +30, +45, +60 min)."""
    preds = []
    s = seq.copy()
    for _ in range(steps):
        p   = model.predict(s, verbose=0)
        val = max(scaler_y.inverse_transform(p)[0][0], 0)
        preds.append(round(float(val), 2))
        new_row    = s[0, -1, :].copy()
        new_row[9] = new_row[8]
        new_row[8] = p[0][0]
        s = np.concatenate([s[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)
    return preds

# ROUTE: Health check
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "message": "BiLSTM Passenger Prediction API is running",
        "endpoints": {
            "POST /auth/register":  "Create account",
            "POST /auth/login":     "Login → get token",
            "GET  /auth/me":        "My profile (protected)",
            "GET  /config":         "Valid dropdown values",
            "POST /predict":        "Single prediction (protected)",
            "POST /forecast":       "Multi-step forecast (protected)",
            "GET  /predictions":    "My prediction history (protected)",
        }
    })

# ROUTE: Config — valid values for frontend dropdowns

@app.route("/config", methods=["GET"])
def get_config():
    """Return valid input values. No auth needed (public)."""
    slots = []
    for h in range(5, 22):
        for m in [0, 30]:
            end_h = h if m == 0 else h + 1
            end_m = 30 if m == 0 else 0
            slots.append(f"{h:02d}:{m:02d}-{end_h:02d}:{end_m:02d}")

    return jsonify({
        "time_slots":      slots,
        "days_of_week":    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        "times_of_day":    config["time_of_day_classes"],
        "peak_options":    ["Peak", "Off-Peak"],
        "bus_stops":       config["busstop_classes"],
        "destinations":    config["destination_classes"],
        "holiday_options": ["No", "Yes"],
        "bus_capacity":    BUS_CAPACITY,
        "seq_len":         SEQ_LEN
    })


# ROUTE: Single Prediction — POST /predict 
@app.route("/predict", methods=["POST"])
@jwt_required()  #  Must send: Authorization: Bearer <token>
def predict():
    try:
        data = request.get_json(force=True)

        #  Validate required fields 
        required = ["time_slot","time_of_day","day_of_week","peak_status",
                    "holiday","bus_stop","destination","lag_t1","lag_t2"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        #  Run model 
        _, seq, bus_cap = run_prediction(data)
        pred_scaled     = model.predict(seq, verbose=0)
        pred_passengers = max(float(scaler_y.inverse_transform(pred_scaled)[0][0]), 0)
        pred_buses      = math.ceil(pred_passengers / bus_cap)
        utilization     = round((pred_passengers / (pred_buses * bus_cap)) * 100, 1)
        capacity_left   = max(pred_buses * bus_cap - int(pred_passengers), 0)
        demand          = bin_demand(pred_passengers)

        # Get the logged-in user's id from the JWT token
        user_id = int(get_jwt_identity())

        # Get current time for week/hour extraction 
        now = datetime.now(timezone.utc)

        # Save prediction to PostgreSQL
        record = Prediction(
            user_id              = user_id,
            time_slot            = data["time_slot"],
            time_of_day          = data["time_of_day"],
            day_of_week          = data["day_of_week"],
            peak_status          = data["peak_status"],
            holiday              = data["holiday"],
            bus_stop             = data["bus_stop"],
            destination          = data["destination"],
            bus_capacity         = bus_cap,
            lag_t1               = float(data["lag_t1"]),
            lag_t2               = float(data["lag_t2"]),
            predicted_passengers = round(pred_passengers, 2),
            buses_required       = pred_buses,
            utilization_percent  = utilization,
            demand_level         = demand,
            capacity_remaining   = capacity_left,
            hour_of_day          = get_hour_from_slot(data["time_slot"]),
            week_number          = now.isocalendar()[1],   
        )
        db.session.add(record)
        db.session.commit()

        #  Return result 
        return jsonify({
            "prediction_id":        record.id,                 
            "predicted_passengers": round(pred_passengers, 2),
            "buses_required":       pred_buses,
            "utilization_percent":  utilization,
            "demand_level":         demand,
            "route":                f"{data['bus_stop']} → {data['destination']}",
            "time_slot":            data["time_slot"],
            "capacity_remaining":   capacity_left,
            "saved":                True
        })

    except Exception as e:
        db.session.rollback()
        print("ERROR IN /predict:")
        traceback.print_exc()   
        return jsonify({"error": str(e)}), 500


# ROUTE: Multi-step Forecast — POST /forecast  
@app.route("/forecast", methods=["POST"])
@jwt_required()
def forecast():
   
    try:
        data = request.get_json(force=True)

        _, seq, bus_cap    = run_prediction(data)
        interval_preds     = forecast_intervals(seq, bus_cap, steps=4)

        intervals = []
        for i, val in enumerate(interval_preds):
            buses = math.ceil(val / bus_cap)
            intervals.append({
                "interval":     f"+{(i+1)*15} min",
                "passengers":   round(val, 2),
                "buses":        buses,
                "demand_level": bin_demand(val)
            })

        return jsonify({
            "route":     f"{data['bus_stop']} → {data['destination']}",
            "time_slot": data["time_slot"],
            "forecast":  intervals
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ROUTE: Get Predictions — GET /predictions 

# GET /predictions/stats — dashboard summary
@app.route("/predictions/stats", methods=["GET"])
@jwt_required()
def prediction_stats():
    from datetime import timedelta
    user_id = int(get_jwt_identity())
    caller  = User.query.get(user_id)

    base = Prediction.query if (caller and caller.role == "admin") \
           else Prediction.query.filter_by(user_id=user_id)

    total_predictions = base.count()
    high_demand_count = base.filter(Prediction.demand_level == "High").count()

    total_buses = db.session.query(
        db.func.sum(Prediction.buses_required)
    ).filter(
        *([] if caller and caller.role == "admin" else [Prediction.user_id == user_id])
    ).scalar() or 0

    avg_utilization = db.session.query(
        db.func.avg(Prediction.utilization_percent)
    ).filter(
        *([] if caller and caller.role == "admin" else [Prediction.user_id == user_id])
    ).scalar() or 0

    DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    by_day = []
    for day in DAYS:
        rows  = base.filter(Prediction.day_of_week == day).all()
        count = len(rows)
        avg_p = round(sum(r.predicted_passengers for r in rows) / count, 1) if count else 0
        by_day.append({"day": day, "count": count, "avg_passengers": avg_p})

    by_hour = []
    for h in range(24):
        rows  = base.filter(Prediction.hour_of_day == h).all()
        count = len(rows)
        avg_p = round(sum(r.predicted_passengers for r in rows) / count, 1) if count else 0
        by_hour.append({"hour": h, "count": count, "avg_passengers": avg_p})

    recent_7_days = []
    for i in range(6, -1, -1):
        day_date = (datetime.now(timezone.utc) - timedelta(days=i)).date()
        rows     = base.filter(db.func.date(Prediction.created_at) == day_date).all()
        count    = len(rows)
        avg_p    = round(sum(r.predicted_passengers for r in rows) / count, 1) if count else 0
        recent_7_days.append({"date": day_date.isoformat(), "count": count, "avg_passengers": avg_p})

    return jsonify({
        "total_predictions": total_predictions,
        "high_demand_count": high_demand_count,
        "total_buses":       int(total_buses),
        "avg_utilization":   round(float(avg_utilization), 1),
        "by_day":            by_day,
        "by_hour":           by_hour,
        "recent_7_days":     recent_7_days,
    }), 200
# RUN THE APP
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
