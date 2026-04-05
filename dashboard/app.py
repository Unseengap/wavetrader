"""
Flask app factory for the WaveTrader Dashboard.
"""
from flask import Flask
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["SECRET_KEY"] = "wavetrader-dashboard-dev"
    app.config["JSON_SORT_KEYS"] = False

    CORS(app)

    # Register blueprints
    from .routes.backtest import backtest_bp
    from .routes.data import data_bp
    from .routes.live import live_bp
    from .routes.pages import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(backtest_bp, url_prefix="/api/backtest")
    app.register_blueprint(data_bp, url_prefix="/api/data")
    app.register_blueprint(live_bp, url_prefix="/api/live")

    return app
