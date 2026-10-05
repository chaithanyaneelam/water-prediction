"""SQLAlchemy tables for WATERNET (section 7 of the spec + auth module)."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model):
    """Account: registers with an email (notifications -> email) or a phone
    number (notifications -> SMS). Password is stored hashed (never plain)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)   # exactly one of email/phone
    phone = db.Column(db.String(20), unique=True)    # E.164-ish, digits with optional +
    password_hash = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(80))
    notify_channel = db.Column(db.String(10), nullable=False, default="outbox")  # email|sms|outbox
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))

    notifications = db.relationship("Notification", back_populates="user",
                                    cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def destination(self) -> str | None:
        """Where notifications go for this user."""
        return self.email if self.notify_channel == "email" else (
            self.phone if self.notify_channel == "sms" else None)


class Notification(db.Model):
    """One prediction message for one user (the notification outbox)."""
    __tablename__ = "notifications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    channel = db.Column(db.String(10), nullable=False)   # email|sms|outbox
    destination = db.Column(db.String(150))
    subject = db.Column(db.String(200))
    body = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="outbox")  # sent|failed|outbox
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    sent_at = db.Column(db.DateTime)

    user = db.relationship("User", back_populates="notifications")


class WaterQualityDataset(db.Model):
    """The training dataset imported from the real Kaggle CSV."""
    __tablename__ = "water_quality_dataset"

    sample_id = db.Column(db.Integer, primary_key=True)
    ph = db.Column(db.Float)
    Hardness = db.Column(db.Float)
    Solids = db.Column(db.Float)
    Chloramines = db.Column(db.Float)
    Sulfate = db.Column(db.Float)
    Conductivity = db.Column(db.Float)
    Organic_carbon = db.Column(db.Float)
    Trihalomethanes = db.Column(db.Float)
    Turbidity = db.Column(db.Float)
    Potability = db.Column(db.Integer)  # 0/1 label (training data only)


class Reading(db.Model):
    """One water test submitted through the app (form or bulk CSV)."""
    __tablename__ = "readings"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.now())
    source = db.Column(db.String(20), nullable=False, default="manual")  # manual|bulk|simulated
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))  # null = anonymous/simulated
    ph = db.Column(db.Float)
    Hardness = db.Column(db.Float, nullable=False)
    Solids = db.Column(db.Float, nullable=False)
    Chloramines = db.Column(db.Float, nullable=False)
    Sulfate = db.Column(db.Float, nullable=False)
    Conductivity = db.Column(db.Float, nullable=False)
    Organic_carbon = db.Column(db.Float, nullable=False)
    Trihalomethanes = db.Column(db.Float, nullable=False)
    Turbidity = db.Column(db.Float, nullable=False)

    prediction = db.relationship(
        "Prediction", back_populates="reading", uselist=False, cascade="all, delete-orphan"
    )


class Prediction(db.Model):
    """Model outputs for a reading (linked 1:1)."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    reading_id = db.Column(db.Integer, db.ForeignKey("readings.id"), nullable=False)
    potability = db.Column(db.Integer, nullable=False)          # 0/1 decision
    probability = db.Column(db.Float, nullable=False)           # P(potable)
    ph_predicted = db.Column(db.Float)                          # None = model not used
    ph_filled_by = db.Column(db.String(20))                     # model|median|none
    is_anomaly = db.Column(db.Boolean, nullable=False, default=False)
    anomaly_score = db.Column(db.Float)
    anomaly_reason = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.now())

    reading = db.relationship("Reading", back_populates="prediction")


class ModelRun(db.Model):
    """Training metadata so the app knows which artifacts are loaded."""
    __tablename__ = "model_runs"

    id = db.Column(db.Integer, primary_key=True)
    run_date = db.Column(db.DateTime, nullable=False, default=db.func.now())
    model_name = db.Column(db.String(50), nullable=False)
    trained_on = db.Column(db.String(20), nullable=False, default="real")  # real|smoke
    metrics_path = db.Column(db.String(200))
    notes = db.Column(db.Text)


class IrrigationReading(db.Model):
    """One irrigation suitability check (rule engine + ML cross-check)."""
    __tablename__ = "irrigation_readings"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=db.func.now())
    source = db.Column(db.String(20), nullable=False, default="manual")
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))  # null = anonymous

    # inputs (mg/L unless noted)
    ph = db.Column(db.Float)
    EC = db.Column(db.Float, nullable=False)   # uS/cm
    TDS = db.Column(db.Float)
    CO3 = db.Column(db.Float)
    HCO3 = db.Column(db.Float, nullable=False)
    Cl = db.Column(db.Float)
    F = db.Column(db.Float)
    NO3 = db.Column(db.Float)
    SO4 = db.Column(db.Float)
    Na = db.Column(db.Float, nullable=False)
    K = db.Column(db.Float)
    Ca = db.Column(db.Float, nullable=False)
    Mg = db.Column(db.Float, nullable=False)
    TH = db.Column(db.Float)

    # rule-engine outputs (deterministic formulas)
    SAR = db.Column(db.Float)
    RSC = db.Column(db.Float)
    ussl_class_rule = db.Column(db.String(8))
    rsc_class_rule = db.Column(db.String(8))
    suitable = db.Column(db.Boolean)
    verdict_notes = db.Column(db.Text)

    # ML cross-check
    ml_ussl_class = db.Column(db.String(8))
    ml_probability = db.Column(db.Float)
