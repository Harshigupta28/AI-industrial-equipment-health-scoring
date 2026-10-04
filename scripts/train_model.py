import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# STEP 1: LOAD SENSOR DATASET
# =========================================================

data = pd.read_csv("data/sensor_data.csv")


# =========================================================
# STEP 2: SELECT INPUT FEATURES
# =========================================================

features = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm"
]

X = data[features]


# =========================================================
# STEP 3: SELECT TARGET
# =========================================================

y = data["status"]


# =========================================================
# STEP 4: TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# STEP 5: CREATE RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# =========================================================
# STEP 6: TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# STEP 7: EVALUATE MODEL
# =========================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model training completed!")
print(f"Test accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# =========================================================
# STEP 8: CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)

print("\nClass order:")
print(model.classes_)


# Create confusion matrix figure
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")

plt.xticks(
    range(len(model.classes_)),
    model.classes_
)

plt.yticks(
    range(len(model.classes_)),
    model.classes_
)


# Add values inside matrix
for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()


# Save confusion matrix
plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()

print(
    "\nConfusion matrix saved to "
    "results/confusion_matrix.png"
)


# =========================================================
# STEP 9: FEATURE IMPORTANCE
# =========================================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")

print(
    feature_importance.to_string(
        index=False
    )
)


# Save feature importance CSV
feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)


# Create feature importance figure
plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Sensor Feature")
plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.tight_layout()


# Save feature importance image
plt.savefig(
    "results/feature_importance.png"
)

plt.close()

print(
    "\nFeature importance saved to "
    "results/feature_importance.png"
)


# =========================================================
# STEP 10: SAVE TRAINED MODEL
# =========================================================

joblib.dump(
    model,
    "models/machine_health_model.pkl"
)

print(
    "\nModel saved successfully to "
    "models/machine_health_model.pkl"
)