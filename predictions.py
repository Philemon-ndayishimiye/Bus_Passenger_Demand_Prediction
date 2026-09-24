

import math
import json
import joblib
import numpy as np
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from database import db
from models import User, Prediction     

#  Load model artifacts once at import time 
from tensorflow.keras.models import load_model as _load_model

_model    = _load_model("best_bilstm_model.h5")
_scaler_X = joblib.load("scaler_X.pkl")
_scaler_y = joblib.load("scaler_y.pkl")
_le_tod   = joblib.load("le_tod.pkl")
_le_dow   = joblib.load("le_dow.pkl")
_le_stop  = joblib.load("le_stop.pkl")
_le_dest  = joblib.load("le_dest.pkl")

with open("model_config.json") as f:
    _cfg = json.load(f)

SEQ_LEN      = _cfg["seq_len"]
BUS_CAPACITY = _cfg["bus_capacity"]

#  Blueprint 
predict_bp = Blueprint("predictions", __name__)


def _parse_hour(time_slot: str) -> float:
    start = time_slot.split("-")[0]
    h, m  = start.split(":")
    return int(h) + int(m) / 60

def _safe_encode(le, value, default=0):
    try:
        return le.transform([value])[0]
    except Exception:
        return default

def _build_sequence(row_features, seq_len):
    row_scaled = _scaler_X.transform([row_features])
    sequence   = np.tile(row_scaled, (seq_len, 1))
    return sequence.reshape(1, seq_len, len(row_features))

def _demand_label(count: float) -> str:
    if count <= 50:   return "Low"
    elif count <= 100: return "Medium"
    else:              return "High"


# POST /predict

@predict_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """
    Run the BiLSTM model and save the result to the DB.

    Expected JSON:
    {
        "time_slot":    "07:00-07:30",
        "time_of_day":  "Morning",
        "day_of_week":  "Monday",
        "peak_status":  "Peak",
        "holiday":      "No",
        "bus_stop":     "Downtown",
        "destination":  "Batsinda",
        "bus_capacity": 70,
        "lag_t1":       85,
        "lag_t2":       72
    }
    """
    data    = request.get_json(force=True)
    user_id = int(get_jwt_identity())

    required = ["time_slot","time_of_day","day_of_week","peak_status",
                "holiday","bus_stop","destination","bus_capacity","lag_t1","lag_t2"]
    for field in required:
        if data.get(field) is None:
            return jsonify({"error": f"'{field}' is required"}), 400

    #  Encode 
    hour_num  = _parse_hour(data["time_slot"])
    tod_enc   = _safe_encode(_le_tod,  data["time_of_day"])
    dow_enc   = _safe_encode(_le_dow,  data["day_of_week"])
    stop_enc  = _safe_encode(_le_stop, data["bus_stop"])
    dest_enc  = _safe_encode(_le_dest, data["destination"])
    peak_enc  = 1 if data["peak_status"] == "Peak" else 0
    hol_enc   = 1 if data["holiday"]     == "Yes"  else 0
    capacity  = int(data["bus_capacity"])

    row_features = [
        hour_num, tod_enc, dow_enc,
        peak_enc, hol_enc, capacity,
        stop_enc, dest_enc,
        float(data["lag_t1"]), float(data["lag_t2"]),
    ]

    #  Predict
    seq              = _build_sequence(row_features, SEQ_LEN)
    pred_scaled      = _model.predict(seq, verbose=0)
    pred_passengers  = max(float(_scaler_y.inverse_transform(pred_scaled)[0][0]), 0)
    buses_required   = math.ceil(pred_passengers / capacity)
    utilization      = round((pred_passengers / (buses_required * capacity)) * 100, 1) if buses_required else 0
    demand_level     = _demand_label(pred_passengers)
    capacity_remain  = max(buses_required * capacity - int(pred_passengers), 0)

    # Derived fields for filtering 
    now         = datetime.utcnow()
    hour_of_day = int(hour_num)
    week_number = now.isocalendar()[1]

    # Save 
    record = Prediction(
        user_id              = user_id,
        time_slot            = data["time_slot"],
        time_of_day          = data["time_of_day"],
        day_of_week          = data["day_of_week"],
        peak_status          = data["peak_status"],
        holiday              = data["holiday"],
        bus_stop             = data["bus_stop"],
        destination          = data["destination"],
        bus_capacity         = capacity,
        lag_t1               = float(data["lag_t1"]),
        lag_t2               = float(data["lag_t2"]),
        predicted_passengers = int(pred_passengers),
        buses_required       = buses_required,
        utilization_percent  = utilization,
        demand_level         = demand_level,
        capacity_remaining   = capacity_remain,
        hour_of_day          = hour_of_day,
        week_number          = week_number,
        created_at           = now,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "prediction_id":       record.id,
        "predicted_passengers": int(pred_passengers),
        "buses_required":       buses_required,
        "utilization_percent":  utilization,
        "demand_level":         demand_level,
        "route":                f"{data['bus_stop']} → {data['destination']}",
        "time_slot":            data["time_slot"],
        "capacity_remaining":   capacity_remain,
        "saved":                True,
    }), 201

# GET /predictions  — with optional filters

@predict_bp.route("/predictions", methods=["GET"])
@jwt_required()
def get_predictions():
    """
    Returns prediction history with optional query-string filters.

    FIX: Each filter is applied only when its query param is present
    and non-empty.  An empty param (?day=) is treated as "no filter".
    """
    user_id = int(get_jwt_identity())
    caller  = User.query.get(user_id)

    # Admins see all predictions; regular users see only their own
    query = Prediction.query if (caller and caller.role == "admin") \
            else Prediction.query.filter_by(user_id=user_id)

    #  Apply filters — only when param is present & non-empty
    day    = (request.args.get("day")    or "").strip()
    hour   = (request.args.get("hour")   or "").strip()
    week   = (request.args.get("week")   or "").strip()
    date   = (request.args.get("date")   or "").strip()
    demand = (request.args.get("demand") or "").strip()
    limit  = int(request.args.get("limit", 200))

    if day:
        query = query.filter(Prediction.day_of_week == day)

    if hour:
        try:
            query = query.filter(Prediction.hour_of_day == int(hour))
        except ValueError:
            pass

    if week:
        try:
            query = query.filter(Prediction.week_number == int(week))
        except ValueError:
            pass

    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            query = query.filter(
                db.func.date(Prediction.created_at) == target_date
            )
        except ValueError:
            pass

    if demand:
        query = query.filter(Prediction.demand_level == demand)

    records = query.order_by(Prediction.created_at.desc()).limit(limit).all()

    return jsonify({
        "total":       query.count(),
        "predictions": [r.to_dict() for r in records],
    }), 200


# GET /predictions/stats  
@predict_bp.route("/predictions/stats", methods=["GET"])
@jwt_required()
def prediction_stats():
    """
    Returns aggregated stats for the dashboard:
      - total_predictions
      - high_demand_count
      - total_buses
      - avg_utilization
      - by_day  (day-of-week breakdown)
      - by_hour (hour-of-day breakdown)
      - recent_7_days (last 7 calendar days)
    """
    user_id = int(get_jwt_identity())
    caller  = User.query.get(user_id)

    base = Prediction.query if (caller and caller.role == "admin") \
           else Prediction.query.filter_by(user_id=user_id)

    total_predictions = base.count()
    high_demand_count = base.filter(Prediction.demand_level == "High").count()
    total_buses       = db.session.query(
        db.func.sum(Prediction.buses_required)
    ).filter(*([] if caller and caller.role == "admin" else [Prediction.user_id == user_id])).scalar() or 0
    avg_utilization   = db.session.query(
        db.func.avg(Prediction.utilization_percent)
    ).filter(*([] if caller and caller.role == "admin" else [Prediction.user_id == user_id])).scalar() or 0

    # by_day
    DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    by_day = []
    for day in DAYS:
        rows  = base.filter(Prediction.day_of_week == day).all()
        count = len(rows)
        avg_p = round(sum(r.predicted_passengers for r in rows) / count, 1) if count else 0
        by_day.append({"day": day, "count": count, "avg_passengers": avg_p})

    # by_hour (0-23)
    by_hour = []
    for h in range(24):
        rows  = base.filter(Prediction.hour_of_day == h).all()
        count = len(rows)
        avg_p = round(sum(r.predicted_passengers for r in rows) / count, 1) if count else 0
        by_hour.append({"hour": h, "count": count, "avg_passengers": avg_p})

    # recent 7 days
    recent_7_days = []
    for i in range(6, -1, -1):
        day_date = (datetime.utcnow() - timedelta(days=i)).date()
        rows     = base.filter(
            db.func.date(Prediction.created_at) == day_date
        ).all()
        count = len(rows)
        avg_p = round(sum(r.predicted_passengers for r in rows) / count, 1) if count else 0
        recent_7_days.append({
            "date":           day_date.isoformat(),
            "count":          count,
            "avg_passengers": avg_p,
        })

    return jsonify({
        "total_predictions": total_predictions,
        "high_demand_count": high_demand_count,
        "total_buses":       int(total_buses),
        "avg_utilization":   round(float(avg_utilization), 1),
        "by_day":            by_day,
        "by_hour":           by_hour,
        "recent_7_days":     recent_7_days,
    }), 200
