"""
app.py — Flask web demo for ASBRS.

Endpoints:
  GET  /                  → serve index.html
  POST /recommend         → run full recommendation pipeline
  GET  /health            → health check
"""

from __future__ import annotations

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Request JSON:
        {"session_items": ["B08N5LNQCX", "B07ZPKBL9V", ...]}
    Response JSON:
        {"recommendations": [...], "intent": str, "rationale": str}
    """
    # Full implementation in Module 06
    data = request.get_json(force=True)
    session_items = data.get("session_items", [])
    if not session_items:
        return jsonify({"error": "session_items must be a non-empty list"}), 400

    # TODO: wire up encoder → retriever → planner → reranker → explainer
    return jsonify({"recommendations": [], "intent": "explore", "rationale": "Demo stub"})


if __name__ == "__main__":
    from config.settings import DEMO
    app.run(host=DEMO["host"], port=DEMO["port"], debug=DEMO["debug"])
