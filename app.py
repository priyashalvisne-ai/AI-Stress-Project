import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('anomaly_model.pkl', 'rb') as f:
    anomaly_model = pickle.load(f)
with open('stress_model.pkl', 'rb') as f:
    stress_model = pickle.load(f)

st.set_page_config(page_title="AI Stress & Biomarker Engine", page_icon="🧠", layout="wide")

st.title("🧠 Multi-Dimensional AI Stress Analyzer")
st.write("Input your daily parameters across physical, academic, and recovery categories for deep AI analysis.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌙 Rest & Recovery")
    sleep_hrs = st.slider("Sleep Duration (Hours)", 2.0, 12.0, 7.0, 0.5)
    sleep_quality = st.select_slider("Sleep Restfulness Rating", options=[1, 2, 3, 4, 5], value=3)
    caffeine_mg = st.number_input("Daily Caffeine Intake (mg)", 0, 800, 150, step=50)

with col2:
    st.subheader("📚 Workload & Cognitive")
    academic_hrs = st.slider("Study / Work Hours", 0.0, 14.0, 6.0, 0.5)
    workload_pressure = st.select_slider("Perceived Work Pressure", options=[1, 2, 3, 4, 5], value=3)
    screen_hrs = st.slider("Non-Work Screen Time (Hours)", 0.0, 14.0, 4.0, 0.5)

with col3:
    st.subheader("🏃 Physical & Social")
    exercise_mins = st.number_input("Physical Activity (Mins/Day)", 0, 180, 30, step=10)
    mood_rating = st.select_slider("Overall Mood Rating", options=[1, 2, 3, 4, 5], value=3)
    social_hrs = st.slider("Socializing (Hours/Week)", 0.0, 30.0, 8.0, 1.0)
    social_quality = st.select_slider("Social Connection Depth", options=[1, 2, 3, 4, 5], value=3)

st.markdown("---")

if st.button("Run AI Factor Analysis"):
    feature_names = [
        'sleep_hrs', 'sleep_quality', 'academic_hrs', 'caffeine_mg', 
        'screen_hrs', 'exercise_mins', 'mood_rating', 'social_hrs', 
        'social_quality', 'workload_pressure'
    ]
    
    user_data = pd.DataFrame([[
        sleep_hrs, sleep_quality, academic_hrs, caffeine_mg,
        screen_hrs, exercise_mins, mood_rating, social_hrs,
        social_quality, workload_pressure
    ]], columns=feature_names)

    # 1. Anomaly Check
    if anomaly_model.predict(user_data)[0] == -1:
        st.error("⚠️ **Contradictory Input Pattern Detected**: The AI flagged this entry as physiologically improbable (e.g., extreme caffeine with maximum sleep quality, or 0 sleep with 14 hours study). Please verify your values.")
    else:
        # 2. Prediction & Probabilities
        prediction = stress_model.predict(user_data)[0]
        probs = stress_model.predict_proba(user_data)[0]
        classes = stress_model.classes_
        prob_dict = dict(zip(classes, probs))
        
        # 3. Calculate Dynamic Stress Index (0-100)
        high_prob = prob_dict.get('High', 0.0)
        mod_prob = prob_dict.get('Moderate', 0.0)
        stress_index = int((high_prob * 100) + (mod_prob * 50))

        st.header(f"Results: Stress Level is **{prediction.upper()}**")
        st.progress(stress_index / 100)
        st.caption(f"Calculated AI Stress Index: **{stress_index}/100** (Confidence: {np.max(probs)*100:.1f}%)")

        # 4. Model Drivers (Explainable AI)
        st.subheader("🔍 Key Model Drivers (Feature Impact)")
        importances = stress_model.feature_importances_
        impact_df = pd.DataFrame({
            'Category': [
                'Sleep Hours', 'Sleep Quality', 'Academic Hours', 'Caffeine Intake',
                'Screen Time', 'Physical Exercise', 'Mood Rating', 'Social Hours',
                'Social Quality', 'Workload Pressure'
            ],
            'Global ML Weight': importances
        }).sort_values(by='Global ML Weight', ascending=False)

        st.bar_chart(impact_df.set_index('Category'))
