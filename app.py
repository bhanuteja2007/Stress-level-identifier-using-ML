import streamlit as st
import pickle
import pandas as pd

# Define the expected feature names
FEATURE_NAMES = ['HRV', 'Sleep_Hours', 'Screen_Time', 'Activity_Level']

# --- 1. Load Model and Scaler ---
try:
    with open('stress_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    model_loaded = True
except FileNotFoundError:
    st.error("Model or Scaler files not found. Run 'python stress_trainer.py' first.")
    model_loaded = False

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Stress Predictor", layout="centered")
st.title("🧠 ML Stress Level Predictor")
st.markdown("---")

if model_loaded:
    st.sidebar.header("Input Physiological & Behavioral Metrics")
    
    # Input Sliders
    hrv = st.sidebar.slider("Heart Rate Variability (HRV)", 20, 100, 50, help="Lower HRV often indicates higher stress.")
    sleep_hours = st.sidebar.slider("Sleep Hours (Last 24h)", 0.0, 12.0, 7.0, help="Total hours of sleep.")
    screen_time = st.sidebar.slider("Daily Screen Time (hours)", 0.0, 10.0, 4.0, help="Total screen time outside of work.")
    activity_level = st.sidebar.slider("Activity Level (Steps/Day)", 100, 15000, 5000, help="Estimated daily step count.")

    # Create DataFrame for the current input
    input_data = pd.DataFrame([[hrv, sleep_hours, screen_time, activity_level]], 
                              columns=FEATURE_NAMES)

    # --- 3. Prediction Logic ---
    if st.button("Predict Stress Level", type="primary"):
        
        # 1. Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # 2. Make the prediction (0 or 1)
        prediction = model.predict(input_scaled)[0]
        
        # 3. Get prediction probabilities
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("### Prediction Result")
        
        if prediction == 1:
            st.error(f"## 🚨 HIGH STRESS PREDICTED")
            st.progress(probability[1], text=f"Confidence: **{probability[1]*100:.2f}%**")
            st.info("Recommendation: The current metrics suggest high stress. Consider prioritizing rest and recovery.")
        else:
            st.success(f"## ✅ LOW STRESS PREDICTED")
            st.progress(probability[0], text=f"Confidence: **{probability[0]*100:.2f}%**")
            st.info("Recommendation: Metrics look good! Maintain your current routine.")

    st.markdown("---")
    st.caption("Input Data Summary:")
    st.dataframe(input_data, use_container_width=True)