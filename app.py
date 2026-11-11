import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

STRESS_META = {
    0: {
        "level": "Low Stress (Balanced)",
        "badge": "success",
        "emoji": "🌱",
        "advice": "Semmaya balance panreenga! Nalla thookkam & workout routine-ah maintain pannunga."
    },
    1: {
        "level": "Moderate Stress (Warning)",
        "badge": "warning",
        "emoji": "⚠️",
        "advice": "Kodi kattuthu! Screen time-ah konjam kuraichi, chinna walk ponga."
    },
    2: {
        "level": "High Burnout Risk (Critical)",
        "badge": "danger",
        "emoji": "🚨",
        "advice": "Danger Zone! Odane break eduthu 7-8 hours nalla thoongunga, caffeine kuraiyunga."
    }
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not os.path.exists(MODEL_PATH):
            return render_template("index.html", error="Model file illai! Mudhalla 'python train_model.py' run pannunga.")

        model = joblib.load(MODEL_PATH)

        study = float(request.form.get("study", 0))
        sleep = float(request.form.get("sleep", 0))
        screen = float(request.form.get("screen", 0))
        caffeine = float(request.form.get("caffeine", 0))
        exercise = float(request.form.get("exercise", 0))

        features = np.array([[study, sleep, screen, caffeine, exercise]])
        pred = int(model.predict(features)[0])
        conf = round(float(np.max(model.predict_proba(features)[0])) * 100, 1)

        meta = STRESS_META.get(pred, {"level": "Unknown", "badge": "secondary", "emoji": "❓", "advice": ""})

        return render_template(
            "index.html",
            level=meta["level"],
            badge=meta["badge"],
            emoji=meta["emoji"],
            advice=meta["advice"],
            confidence=conf,
            study=study,
            sleep=sleep,
            screen=screen,
            caffeine=caffeine,
            exercise=exercise
        )
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
