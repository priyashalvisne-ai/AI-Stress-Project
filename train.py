import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, IsolationForest

np.random.seed(42)
n = 1500

sleep_hrs = np.random.uniform(3, 10, n)
sleep_quality = np.random.randint(1, 6, n)
academic_hrs = np.random.uniform(1, 10, n)
caffeine_mg = np.random.uniform(0, 600, n)
screen_hrs = np.random.uniform(1, 14, n)
exercise_mins = np.random.uniform(0, 150, n)
mood_rating = np.random.randint(1, 6, n)
social_hrs = np.random.uniform(0, 25, n)
social_quality = np.random.randint(1, 6, n)
workload_pressure = np.random.randint(1, 6, n)

stress_score = (
    (10 - sleep_hrs) * 1.8 +
    (6 - sleep_quality) * 1.5 +
    academic_hrs * 1.2 +
    (caffeine_mg / 100) * 0.8 +
    (screen_hrs / 2) * 0.9 +
    workload_pressure * 2.0 -
    (exercise_mins / 30) * 1.2 -
    (mood_rating) * 1.5 -
    (social_hrs / 5) * 0.5 -
    (social_quality) * 1.0
)

labels = ['Low' if s < 8 else 'Moderate' if s < 18 else 'High' for s in stress_score]

df = pd.DataFrame({
    'sleep_hrs': sleep_hrs, 'sleep_quality': sleep_quality,
    'academic_hrs': academic_hrs, 'caffeine_mg': caffeine_mg,
    'screen_hrs': screen_hrs, 'exercise_mins': exercise_mins,
    'mood_rating': mood_rating, 'social_hrs': social_hrs,
    'social_quality': social_quality, 'workload_pressure': workload_pressure,
    'stress_level': labels
})

X = df.drop(columns=['stress_level'])
y = df['stress_level']

anomaly_model = IsolationForest(contamination=0.04, random_state=42).fit(X)
stress_model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42).fit(X, y)

with open('anomaly_model.pkl', 'wb') as f:
    pickle.dump(anomaly_model, f)
with open('stress_model.pkl', 'wb') as f:
    pickle.dump(stress_model, f)

print("✅ Training complete. PKL files created!")
