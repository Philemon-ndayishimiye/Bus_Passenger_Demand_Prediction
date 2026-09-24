

from datetime import datetime
from database import db


# USER
class User(db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(120), nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    role       = db.Column(db.String(20), nullable=False, default="user")  
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship 
    predictions = db.relationship("Prediction", backref="user", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "role":       self.role,
            "created_at": self.created_at.isoformat(),
        }

    def to_dict_with_predictions(self):
        """Extended dict that includes the user's prediction count."""
        d = self.to_dict()
        d["prediction_count"] = self.predictions.count()
        return d


# TOKEN BLOCKLIST (for logout)
class TokenBlocklist(db.Model):
    __tablename__ = "token_blocklist"

    id         = db.Column(db.Integer, primary_key=True)
    jti        = db.Column(db.String(36), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# PREDICTION
class Prediction(db.Model):
    __tablename__ = "predictions"

    id                   = db.Column(db.Integer, primary_key=True)
    user_id              = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # Input fields
    time_slot            = db.Column(db.String(20), nullable=False)
    time_of_day          = db.Column(db.String(20), nullable=False)
    day_of_week          = db.Column(db.String(20), nullable=False)
    peak_status          = db.Column(db.String(20), nullable=False)
    holiday              = db.Column(db.String(5),  nullable=False)
    bus_stop             = db.Column(db.String(100), nullable=False)
    destination          = db.Column(db.String(100), nullable=False)
    bus_capacity         = db.Column(db.Integer, nullable=False)
    lag_t1               = db.Column(db.Float, nullable=False)
    lag_t2               = db.Column(db.Float, nullable=False)

    # Output fields
    predicted_passengers = db.Column(db.Integer, nullable=False)
    buses_required       = db.Column(db.Integer, nullable=False)
    utilization_percent  = db.Column(db.Float,   nullable=False)
    demand_level         = db.Column(db.String(10), nullable=False)  
    capacity_remaining   = db.Column(db.Integer, nullable=False)

    # Derived fields for fast filtering
    hour_of_day          = db.Column(db.Integer, nullable=False, default=0)   
    week_number          = db.Column(db.Integer, nullable=False, default=1)  

    created_at           = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":                    self.id,
            "user_id":               self.user_id,
            "time_slot":             self.time_slot,
            "time_of_day":           self.time_of_day,
            "day_of_week":           self.day_of_week,
            "peak_status":           self.peak_status,
            "holiday":               self.holiday,
            "bus_stop":              self.bus_stop,
            "destination":           self.destination,
            "bus_capacity":          self.bus_capacity,
            "lag_t1":                self.lag_t1,
            "lag_t2":                self.lag_t2,
            "predicted_passengers":  self.predicted_passengers,
            "buses_required":        self.buses_required,
            "utilization_percent":   self.utilization_percent,
            "demand_level":          self.demand_level,
            "capacity_remaining":    self.capacity_remaining,
            "hour_of_day":           self.hour_of_day,
            "week_number":           self.week_number,
            "created_at":            self.created_at.isoformat(),
        }
