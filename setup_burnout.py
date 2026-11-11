import os

files = {
    # 1. requirements.txt
    "requirements.txt": """flask
scikit-learn
joblib
numpy
""",
    # 2. train_model.py (Actual ML Training with Scikit-learn)
    "train_model.py": """import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Features: [Study Hours, Sleep Hours, Screen Time, Coffee/Tea Cups, Physical Activity (mins)]
# Classes:
# 0 -> Low Stress / Balanced Life
# 1 -> Moderate Stress / Warning Zone
# 2 -> High Burnout Risk / Danger Zone

X = np.array([
    # [Study, Sleep, Screen, Caffeine, Exercise]
    [4, 8, 3, 1, 45],
    [5, 7.5, 4, 1, 30],
    [3, 8, 2, 0, 60],
    [4.5, 7, 3.5, 1, 40],   # Low Stress (0)

    [7, 6, 6, 3, 20],
    [8, 5.5, 7, 2, 15],
    [6.5, 6, 5, 2, 25],
    [7.5, 5, 6.5, 3, 10],   # Moderate Stress (1)

    [11, 4, 9, 5, 0],
    [12, 3.5, 10, 4, 0],
    [10, 4.5, 8, 4, 5],
    [13, 3, 11, 5, 0],      # High Burnout (2)

    [4, 8.5, 3, 0, 50],
    [7, 5.5, 6, 3, 15],
    [11.5, 4, 9.5, 4, 0]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2])

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=30, random_state=42)
clf.fit(X, y)

# Save ML model
joblib.dump(clf, "model.joblib")
print("ML Model trained successfully! 'model.joblib' file generated.")
""",
    # 3. app.py (Flask Backend)
    "app.py": """import os
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
""",
    # 4. vercel.json
    "vercel.json": """{
  "version": 2,
  "builds": [{"src": "app.py", "use": "@vercel/python"}],
  "routes": [{"src": "/(.*)", "dest": "app.py"}]
}
""",
    # 5. .gitignore
    ".gitignore": """venv/
__pycache__/
*.pyc
.env
""",
    # 6. templates/index.html (Modern Dark Mode UI)
    "templates/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Student Burnout Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0b0f19; color: #f3f4f6; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 25px; }
        .card { background: #111827; border: 1px solid #1f2937; border-radius: 16px; width: 100%; max-width: 480px; padding: 28px; box-shadow: 0 15px 35px rgba(0,0,0,0.6); }
        .badge { display: inline-block; background: #6366f1; color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }
        h1 { font-size: 22px; font-weight: 700; color: #f9fafb; margin-bottom: 4px; }
        p.sub { font-size: 13px; color: #9ca3af; margin-bottom: 22px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }
        .field { display: flex; flex-direction: column; }
        .full-width { grid-column: span 2; }
        label { font-size: 12px; font-weight: 600; color: #d1d5db; margin-bottom: 6px; }
        input { background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 10px 12px; color: #fff; font-size: 14px; outline: none; transition: border 0.2s; }
        input:focus { border-color: #6366f1; }
        button { width: 100%; background: #6366f1; color: white; border: none; border-radius: 8px; padding: 13px; font-size: 15px; font-weight: 700; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #4f46e5; }
        .output { margin-top: 22px; padding: 16px; border-radius: 10px; text-align: center; border: 1px solid transparent; }
        .output.success { background: rgba(16, 185, 129, 0.12); border-color: #10b981; color: #34d399; }
        .output.warning { background: rgba(245, 158, 11, 0.12); border-color: #f59e0b; color: #fbbf24; }
        .output.danger { background: rgba(239, 68, 68, 0.12); border-color: #ef4444; color: #f87171; }
        .res-level { font-size: 18px; font-weight: 800; margin-bottom: 4px; }
        .res-advice { font-size: 13px; margin-bottom: 8px; opacity: 0.95; }
        .res-conf { font-size: 12px; font-weight: 600; opacity: 0.8; }
        .footer { text-align: center; margin-top: 20px; font-size: 11px; color: #6b7280; }
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">ML Classifier • Random Forest</span>
        <h1>AI Student Burnout Detector</h1>
        <p class="sub">Lifestyle metrics vechi stress & burnout risk-ah predict pannum.</p>

        <form action="/predict" method="POST">
            <div class="grid">
                <div class="field">
                    <label>Study Time (Hours/day)</label>
                    <input type="number" step="any" name="study" placeholder="e.g. 6" value="{{ study if study is defined else '' }}" required>
                </div>
                <div class="field">
                    <label>Sleep Duration (Hours)</label>
                    <input type="number" step="any" name="sleep" placeholder="e.g. 7" value="{{ sleep if sleep is defined else '' }}" required>
                </div>
                <div class="field">
                    <label>Screen Time (Hours/day)</label>
                    <input type="number" step="any" name="screen" placeholder="e.g. 5" value="{{ screen if screen is defined else '' }}" required>
                </div>
                <div class="field">
                    <label>Caffeine (Cups/day)</label>
                    <input type="number" step="any" name="caffeine" placeholder="e.g. 2" value="{{ caffeine if caffeine is defined else '' }}" required>
                </div>
                <div class="field full-width">
                    <label>Exercise / Walk (Minutes/day)</label>
                    <input type="number" step="any" name="exercise" placeholder="e.g. 30" value="{{ exercise if exercise is defined else '' }}" required>
                </div>
            </div>
            <button type="submit">Predict Burnout Level</button>
        </form>

        {% if level %}
        <div class="output {{ badge }}">
            <div class="res-level">{{ emoji }} {{ level }}</div>
            <div class="res-advice">{{ advice }}</div>
            <div class="res-conf">Model Confidence Score: {{ confidence }}%</div>
        </div>
        {% elif error %}
        <div class="output danger">
            <div class="res-level">Error</div>
            <div class="res-advice">{{ error }}</div>
        </div>
        {% endif %}

        <div class="footer">DevOps Ready • Flask + Scikit-Learn + Vercel</div>
    </div>
</body>
</html>
""",
}

for path, content in files.items():
  dir_name = os.path.dirname(path)
  if dir_name:
    os.makedirs(dir_name, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(
    "Super! Ella files-um (app.py, train_model.py, templates/index.html,"
    " requirements.txt, vercel.json) auto-generate aayiduchu!"
)