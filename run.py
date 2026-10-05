"""WATERNET entry point: python run.py"""
from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    import os
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
