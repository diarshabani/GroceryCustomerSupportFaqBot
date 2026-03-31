from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from chatbot import get_answer
from inventory_utils import DISPLAY_COLUMNS, load_inventory_records


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Grocery_Inventory_and_Sales_Dataset.csv"
MAX_HISTORY_TURNS = 10

app = Flask(__name__)

INVENTORY_DATA = load_inventory_records(CSV_PATH)


def _normalize_chat_history(raw_history: Any) -> list[dict[str, str]]:
    """Keep only recent user and assistant turns for follow-up resolution."""
    if not isinstance(raw_history, list):
        return []

    history: list[dict[str, str]] = []
    for item in raw_history[-MAX_HISTORY_TURNS * 2 :]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})

    return history


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        inventory_data=INVENTORY_DATA,
        display_columns=DISPLAY_COLUMNS,
    )


@app.post("/chat")
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    history = _normalize_chat_history(payload.get("history", []))

    if not question:
        return jsonify({"answer": "Please enter a question."})

    try:
        answer = get_answer(question, history=history)
    except Exception:
        answer = (
            "I'm sorry, but I couldn't reach the grocery FAQ service right now. "
            "Please try again in a moment."
        )

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
