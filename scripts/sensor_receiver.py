import os
import ssl
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd
import joblib
import paho.mqtt.client as mqtt
from dotenv import load_dotenv


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "machine_health_model.pkl"
DB_PATH = BASE_DIR / "database" / "machine_health.db"
ENV_PATH = BASE_DIR / ".env"


# =========================================================
# LOAD MQTT CONFIGURATION
# =========================================================

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)

HOST = os.getenv("MQTT_HOST")
PORT = int(os.getenv("MQTT_PORT", "8883"))
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")

TOPIC = "factory/machine1/sensor-data"


# =========================================================
# SENSOR FEATURES
# =========================================================

FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm"
]


# =========================================================
# LOAD ML MODEL
# =========================================================

model = joblib.load(MODEL_PATH)

print("ML model loaded successfully!")


# =========================================================
# CHECK MQTT CONFIGURATION
# =========================================================

if not all([HOST, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing MQTT settings. Check your .env file."
    )


# =========================================================
# SAVE READING TO SQLITE
# =========================================================

def save_reading(sensor_data, predicted_status):

    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sensor_readings (
                temperature,
                vibration,
                pressure,
                current,
                rpm,
                predicted_status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sensor_data["temperature"],
                sensor_data["vibration"],
                sensor_data["pressure"],
                sensor_data["current"],
                sensor_data["rpm"],
                str(predicted_status)
            )
        )

        conn.commit()

    print("Reading and prediction saved to database!")


# =========================================================
# MQTT CONNECT CALLBACK
# =========================================================

def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties
):

    if reason_code.is_failure:

        print(
            "Connection failed:",
            reason_code
        )

        client.disconnect()
        return

    print("Connected to HiveMQ Cloud!")

    result, mid = client.subscribe(
        TOPIC,
        qos=1
    )

    if result == mqtt.MQTT_ERR_SUCCESS:

        print(
            "Subscribed to:",
            TOPIC
        )

    else:

        print(
            "Subscription failed:",
            result
        )

        client.disconnect()


# =========================================================
# MQTT MESSAGE CALLBACK
# =========================================================

def on_message(
    client,
    userdata,
    msg
):

    try:

        # Convert MQTT payload to JSON
        received_data = json.loads(
            msg.payload.decode("utf-8")
        )

        if not isinstance(received_data, dict):

            print(
                "Invalid message: "
                "expected a JSON object."
            )

            return


        # Check required fields
        missing = [
            field
            for field in FEATURES
            if field not in received_data
        ]

        if missing:

            print(
                "Missing sensor fields:",
                missing
            )

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

            if not math.isfinite(
                values[field]
            ):

                raise ValueError(
                    f"{field} must be finite."
                )


        # Prepare model input
        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )


        print("\nReceived sensor readings:")

        print(
            input_data.to_string(
                index=False
            )
        )


        # ML prediction
        prediction = model.predict(
            input_data
        )[0]


        print(
            "\nPredicted machine status:",
            prediction
        )


        # Save to SQLite
        save_reading(
            values,
            prediction
        )


        print(
            "MQTT + ML + SQLite integration successful!"
        )


    except (
        UnicodeDecodeError,
        json.JSONDecodeError
    ):

        print(
            "Error: Received message "
            "is not valid JSON."
        )


    except (
        ValueError,
        TypeError,
        KeyError
    ) as error:

        print(
            "Invalid sensor data:",
            error
        )


    except sqlite3.Error as error:

        print(
            "Database error:",
            error
        )


# =========================================================
# CREATE MQTT CLIENT
# =========================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="machine-health-receiver"
)


client.username_pw_set(
    USERNAME,
    PASSWORD
)


client.tls_set(
    cert_reqs=ssl.CERT_REQUIRED
)


# Register callbacks
client.on_connect = on_connect
client.on_message = on_message


# =========================================================
# START RECEIVER
# =========================================================

print(
    "Starting MQTT receiver..."
)

try:

    client.connect(
        HOST,
        PORT,
        keepalive=60
    )

    client.loop_forever()


except KeyboardInterrupt:

    print(
        "\nMQTT receiver stopped."
    )


finally:

    client.disconnect()