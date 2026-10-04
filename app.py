import streamlit as st
import pandas as pd
import joblib
import sqlite3
from pathlib import Path

from streamlit_autorefresh import st_autorefresh
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "machine_health_model.pkl"
DATA_PATH = BASE_DIR / "data" / "sensor_data.csv"
DB_PATH = BASE_DIR / "database" / "machine_health.db"
CONFUSION_MATRIX_PATH = BASE_DIR / "results" / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.png"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Machine Health Monitor",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=5000,
    key="machine_health_refresh"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("🏭 AI-Based Industrial Equipment Health Scoring")

st.write(
    "Monitor machine health using sensor readings, "
    "Machine Learning, MQTT and SQLite."
)

st.caption("🔄 Dashboard auto-refreshes every 5 seconds.")

st.divider()


# =========================================================
# LIVE MACHINE SUMMARY
# =========================================================

st.subheader("📡 Live Machine Status")

try:

    with sqlite3.connect(DB_PATH) as conn:

        latest_data = pd.read_sql_query(
            """
            SELECT
                temperature,
                vibration,
                pressure,
                current,
                rpm,
                predicted_status,
                timestamp
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT 1
            """,
            conn
        )

        total_readings = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total
            FROM sensor_readings
            """,
            conn
        )["total"].iloc[0]

    if not latest_data.empty:

        latest = latest_data.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Latest Status",
                latest["predicted_status"]
            )

        with col2:
            st.metric(
                "Temperature",
                f"{latest['temperature']:.2f} °C"
            )

        with col3:
            st.metric(
                "Vibration",
                f"{latest['vibration']:.2f} mm/s"
            )

        with col4:
            st.metric(
                "Total Readings",
                int(total_readings)
            )

        if latest["predicted_status"] == "Healthy":

            st.success("🟢 Machine Status: Healthy")

        elif latest["predicted_status"] == "Warning":

            st.warning("🟡 Machine Status: Warning")

        else:

            st.error("🔴 Machine Status: Critical")

        st.caption(
            f"Last sensor update: {latest['timestamp']}"
        )

    else:

        st.info("Waiting for sensor data...")

except sqlite3.Error as error:

    st.error(
        f"Could not load live machine status: {error}"
    )


st.divider()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("🤖 ML Model Performance")

try:

    data = pd.read_csv(DATA_PATH)

    features = [
        "temperature",
        "vibration",
        "pressure",
        "current",
        "rpm"
    ]

    X = data[features]
    y = data["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True
    )

    # Model metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Test Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:
        st.metric(
            "Test Samples",
            len(X_test)
        )

    with col3:
        st.metric(
            "Training Samples",
            len(X_train)
        )

    # Classification report
    st.write("### Classification Report")

    report_data = pd.DataFrame(report).transpose()

    st.dataframe(
        report_data.round(3),
        use_container_width=True
    )

    # Confusion Matrix + Feature Importance
    col1, col2 = st.columns(2)

    with col1:

        st.write("### Confusion Matrix")

        if CONFUSION_MATRIX_PATH.exists():

            st.image(
                str(CONFUSION_MATRIX_PATH),
                use_container_width=True
            )

        else:

            st.info(
                "Confusion matrix image not found."
            )

    with col2:

        st.write("### Feature Importance")

        if FEATURE_IMPORTANCE_PATH.exists():

            st.image(
                str(FEATURE_IMPORTANCE_PATH),
                use_container_width=True
            )

        else:

            st.info(
                "Feature importance image not found."
            )

except Exception as error:

    st.error(
        f"Could not load model evaluation: {error}"
    )


st.divider()


# =========================================================
# MANUAL MACHINE HEALTH CHECK
# =========================================================

st.subheader("🔍 Manual Machine Health Check")

st.write(
    "Enter sensor values to test the trained machine-health model."
)

col1, col2 = st.columns(2)


with col1:

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0,
        max_value=150.0,
        value=60.0,
        step=0.5
    )

    vibration = st.number_input(
        "📳 Vibration (mm/s)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.1
    )

    pressure = st.number_input(
        "⏱️ Pressure (bar)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )


with col2:

    current = st.number_input(
        "⚡ Current (A)",
        min_value=0.0,
        max_value=50.0,
        value=12.0,
        step=0.1
    )

    rpm = st.number_input(
        "🔄 RPM",
        min_value=0.0,
        max_value=3000.0,
        value=1500.0,
        step=50.0
    )


# Prepare sensor data
machine = pd.DataFrame([{
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "current": current,
    "rpm": rpm
}])


# =========================================================
# MACHINE HEALTH PREDICTION
# =========================================================

if st.button(
    "🚀 Check Machine Health",
    type="primary",
    use_container_width=True
):

    status = model.predict(machine)[0]

    # Illustrative score only
    score_map = {
        "Healthy": 90,
        "Warning": 60,
        "Critical": 25
    }

    score = score_map[status]

    st.divider()

    st.subheader("📋 Machine Health Report")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Illustrative Health Score",
            f"{score}/100"
        )

        st.progress(score / 100)

    with col2:

        if status == "Healthy":

            st.success(
                "🟢 HEALTHY\n\n"
                "The model classifies this machine "
                "reading as Healthy."
            )

        elif status == "Warning":

            st.warning(
                "🟡 WARNING\n\n"
                "The model classifies this machine "
                "reading as Warning."
            )

        else:

            st.error(
                "🔴 CRITICAL\n\n"
                "The model classifies this machine "
                "reading as Critical."
            )

    st.write("### Sensor Input Used for Prediction")

    display_machine = machine.copy()

    display_machine.columns = [
        "Temperature (°C)",
        "Vibration (mm/s)",
        "Pressure (bar)",
        "Current (A)",
        "RPM"
    ]

    st.dataframe(
        display_machine,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Demo only: the model uses simulated data and "
        "rule-generated labels. The score is illustrative, "
        "not a validated measure of real machine health."
    )


st.divider()


# =========================================================
# DATABASE HISTORY
# =========================================================

st.subheader("📊 Sensor Reading History")

st.write(
    "Latest sensor readings and ML predictions saved in SQLite."
)

try:

    with sqlite3.connect(DB_PATH) as conn:

        history = pd.read_sql_query(
            """
            SELECT
                id AS 'Record ID',
                temperature AS 'Temperature (°C)',
                vibration AS 'Vibration (mm/s)',
                pressure AS 'Pressure (bar)',
                current AS 'Current (A)',
                rpm AS 'RPM',
                predicted_status AS 'Predicted Status',
                timestamp AS 'Timestamp'
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT 20
            """,
            conn
        )

    if not history.empty:

        st.metric(
            "Saved Records Shown",
            len(history)
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        # Sensor trends
        st.divider()

        st.subheader("📈 Sensor Trends")

        chart_data = history.sort_values(
            "Record ID"
        )

        st.write("Temperature Trend")

        st.line_chart(
            chart_data.set_index(
                "Timestamp"
            )[["Temperature (°C)"]]
        )

        st.write("Vibration Trend")

        st.line_chart(
            chart_data.set_index(
                "Timestamp"
            )[["Vibration (mm/s)"]]
        )

        # Status distribution
        st.subheader(
            "📊 Machine Status Distribution"
        )

        all_statuses = pd.DataFrame({
            "Status": [
                "Healthy",
                "Warning",
                "Critical"
            ]
        })

        actual_counts = (
            history["Predicted Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )

        status_counts = all_statuses.merge(
            actual_counts,
            on="Status",
            how="left"
        )

        status_counts["Count"] = (
            status_counts["Count"]
            .fillna(0)
            .astype(int)
        )

        st.bar_chart(
            status_counts.set_index("Status")
        )

    else:

        st.info(
            "No sensor readings have been saved yet."
        )

except sqlite3.Error as error:

    st.error(
        f"Could not load database history: {error}"
    )