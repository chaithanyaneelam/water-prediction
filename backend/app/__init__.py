"""WATERNET Flask application factory."""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

from backend.app.models import db
from backend.app.services.model_service import model_service


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)
    CORS(app)

    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
        "DATABASE_URL",
        # Default: SQLite file at the project root (see backend/ml/config.py).
        "sqlite:///" + os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "waternet.db"
        ),
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")

    db.init_app(app)
    with app.app_context():
        db.create_all()  # tables: water_quality_dataset, readings, predictions, model_runs

    # Models are loaded ONCE at startup. If artifacts are missing the site still
    # runs; prediction endpoints answer 503 with a hint to run training.
    with app.app_context():
        try:
            status = model_service.load()
            print(f"[startup] models loaded: {status}")
        except FileNotFoundError:
            print("[startup] WARNING: saved models not found - prediction endpoints "
                  "will return 503. Run: python -m backend.ml.train_all")

    from backend.app.blueprints.pages import bp as pages_bp
    from backend.app.blueprints.predict_api import bp as predict_bp
    from backend.app.blueprints.readings_api import bp as readings_bp
    from backend.app.blueprints.stats_api import bp as stats_bp
    from backend.app.blueprints.models_api import bp as models_bp
    from backend.app.blueprints.dataset_api import bp as dataset_bp
    from backend.app.blueprints.irrigation_api import bp as irrigation_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(readings_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(irrigation_bp)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "models_loaded": model_service.loaded})

    return app
