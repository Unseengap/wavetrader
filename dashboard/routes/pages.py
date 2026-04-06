"""
Page routes — serves the dashboard HTML.
"""
from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return render_template("index.html")


@pages_bp.route("/logs")
def logs():
    return render_template("logs.html")
