import os
import ssl
import json
import math
import sqlite3
import pandas as pd
import joblib
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load MQTT settings from .env
load_dotenv(override=True)

HOST = os.getenv("MQTT_HOST")
PORT = int(os.getenv("MQTT_PORT", "8883"))
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")

TOPIC = "factory/machine1/sensor-data"
DB_NAME = "machine_health.db"

FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm"
]

# Load trained ML model
model = joblib.load("machine_health_model.pkl")
print("ML model loaded successfully!")

# Check MQTT configuration without exposing secrets
if not all([HOST, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing MQTT settings. Check your .env file."
    )


# Save sensor readings and prediction to SQLite
def save_reading(sensor_data, predicted_status):
    conn = sqlite3.connect(DB_NAME)

    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sensor_readings (
                temperature,
                vibration,
                pressure,
                current,
                rpm,
                predicted_status
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sensor_data["temperature"],
            sensor_data["vibration"],
            sensor_data["pressure"],
            sensor_data["current"],
            sensor_data["rpm"],
            str(predicted_status)
        ))

        conn.commit()
        print("Reading and prediction saved to database!")

    finally:
        conn.close()


# Configure MQTT client
client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="machine-health-test"
)

client.username_pw_set(USERNAME, PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED)


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print("Connection failed:", reason_code)
        client.disconnect()
        return

    print("Connected to HiveMQ Cloud!")

    result, mid = client.subscribe(TOPIC)

    if result == mqtt.MQTT_ERR_SUCCESS:
        print("Subscription requested:", TOPIC)
    else:
        print("Subscription request failed:", result)
        client.disconnect()


def on_subscribe(client, userdata, mid, reason_code_list, properties):
    if any(code.is_failure for code in reason_code_list):
        print("Subscription failed:", reason_code_list)
        client.disconnect()
        return

    print("Subscription confirmed!")

    # Sample sensor readings for this test
    sensor_data = {
        "temperature": 65.5,
        "vibration": 3.2,
        "pressure": 5.1,
        "current": 12.4,
        "rpm": 1500
    }

    payload = json.dumps(sensor_data)
    result = client.publish(TOPIC, payload)

    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("Sensor data published:", payload)
    else:
        print("Publish failed:", result.rc)
        client.disconnect()


def on_message(client, userdata, msg):
    try:
        received_data = json.loads(
            msg.payload.decode("utf-8")
        )

        if not isinstance(received_data, dict):
            print("Invalid message: expected a JSON object.")
            return

        missing = [
            field for field in FEATURES
            if field not in received_data
        ]

        if missing:
            print("Missing sensor fields:", missing)
            return

        # Validate sensor values
        values = {}

        for field in FEATURES:
            value = received_data[field]

            if isinstance(value, bool):
                raise ValueError(
                    f"{field} must be numeric."
                )

            values[field] = float(value)

            if not math.isfinite(values[field]):
                raise ValueError(
                    f"{field} must be finite."
                )

        # Prepare model input
        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        print("\nReceived sensor readings:")
        print(input_data.to_string(index=False))

        # Predict machine status
        prediction = model.predict(input_data)[0]

        print("\nPredicted machine status:", prediction)

        # Save reading and prediction to database
        save_reading(values, prediction)

        print("MQTT + ML + SQLite integration successful!")

    except (UnicodeDecodeError, json.JSONDecodeError):
        print("Error: Received message is not valid JSON.")

    except (ValueError, TypeError, KeyError) as error:
        print("Invalid sensor data:", error)

    except sqlite3.Error as error:
        print("Database error:", error)

    finally:
        # This is a one-message test
        client.disconnect()


client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message

print("Connecting to HiveMQ Cloud...")

client.connect(HOST, PORT, keepalive=60)
client.loop_forever()
