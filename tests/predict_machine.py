import pandas as pd
import joblib

# Load our trained model
model = joblib.load("machine_health_model.pkl")

# Example readings from a machine
new_machine = pd.DataFrame([{
    "temperature": 60,
    "vibration": 3,
    "pressure": 5,
    "current": 12,
    "rpm": 1500
}])

# Predict the machine status
prediction = model.predict(new_machine)

print("New machine readings:")
print(new_machine)

print("\nPredicted machine status:", prediction[0])