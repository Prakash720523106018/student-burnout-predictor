import numpy as np
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
