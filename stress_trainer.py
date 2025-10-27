import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Data Simulation (Replace with your actual data loading) ---
def create_synthetic_stress_data(n_samples=1000):
    """Generates synthetic data for stress level prediction."""
    np.random.seed(42)
    
    # Features (Input)
    hrv = np.random.normal(50, 10, n_samples)
    sleep_hrs = np.random.normal(7, 1.5, n_samples)
    screen_time = np.random.normal(4, 2, n_samples)
    activity_level = np.random.normal(5000, 2000, n_samples)

    # Simple rule for Target (Stress Level: 0=Low, 1=High)
    stress_probability = (
        (1 / hrv) * 10 
        + (1 / sleep_hrs) * 5 
        + screen_time * 0.5 
        - activity_level * 0.0001
    )
    
    stress_probability = (stress_probability - stress_probability.min()) / (stress_probability.max() - stress_probability.min())
    stress_level = (stress_probability > 0.6).astype(int) 

    data = pd.DataFrame({
        'HRV': hrv,
        'Sleep_Hours': sleep_hrs,
        'Screen_Time': screen_time,
        'Activity_Level': activity_level,
        'Stress_Level': stress_level
    })
    
    return data

df = create_synthetic_stress_data(n_samples=1000)
print(f"Data Loaded. Shape: {df.shape}")

# --- 2. Preprocessing and Splitting ---
X = df.drop('Stress_Level', axis=1)
y = df['Stress_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and Fit Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Model Training ---
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Training complete.")

# --- 4. Evaluation ---
y_pred = model.predict(X_test_scaled)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Low Stress (0)', 'High Stress (1)']))

# --- 5. Saving the Model and Scaler ---
MODEL_FILE = 'stress_model.pkl'
SCALER_FILE = 'scaler.pkl'

with open(MODEL_FILE, 'wb') as file:
    pickle.dump(model, file)
    print(f"\nModel saved successfully as: {MODEL_FILE}")

with open(SCALER_FILE, 'wb') as file:
    pickle.dump(scaler, file)
    print(f"Scaler saved successfully as: {SCALER_FILE}")