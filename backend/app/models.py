"""SQLAlchemy tables for WATERNET (section 7 of the spec)."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


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
