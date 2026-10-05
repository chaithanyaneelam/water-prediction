"""HTML pages (Flask serves the frontend - one command starts the site)."""
from flask import Blueprint, render_template

bp = Blueprint("pages", __name__)


@bp.get("/")
def dashboard():
    return render_template("dashboard.html")


@bp.get("/predict")
def predict():
    return render_template("predict.html")


@bp.get("/bulk")
def bulk():
    return render_template("bulk.html")


@bp.get("/history")
def history():
    return render_template("history.html")


@bp.get("/comparison")
def comparison():
    return render_template("comparison.html")


@bp.get("/explorer")
def explorer():
    return render_template("explorer.html")


@bp.get("/irrigation")
def irrigation():
    return render_template("irrigation.html")
