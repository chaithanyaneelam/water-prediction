"""WATERNET Flask application factory (placeholder for STEP 0)."""
from flask import Flask, render_template
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return "<h1>WATERNET - placeholder</h1><p>Backend skeleton is alive. Building modules next...</p>"

    return app
