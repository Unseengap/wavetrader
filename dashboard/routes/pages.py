"""
Page routes — serves the dashboard HTML.
"""
from flask import Blueprint, redirect, render_template, url_for

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return redirect(url_for("pages.live"))


@pages_bp.route("/live")
def live():
    return render_template("live.html")


@pages_bp.route("/backtest")
def backtest():
    return render_template("backtest.html")


@pages_bp.route("/logs")
def logs():
    return render_template("logs.html")
