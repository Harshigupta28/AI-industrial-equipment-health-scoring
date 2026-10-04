import os
import ssl
import json
import time
import random
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
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
# CHECK CONFIGURATION
# =========================================================

if not all([HOST, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing MQTT settings. Check your .env file."
    )


# =========================================================
# GENERATE SENSOR DATA
# =========================================================

def generate_sensor_data():

    return {
        "temperature": round(
            random.uniform(50, 95),
            2
        ),

        "vibration": round(
            random.uniform(2, 11),
            2
        ),

        "pressure": round(
            random.uniform(4, 8),
            2
        ),

        "current": round(
            random.uniform(8, 20),
            2
        ),

        "rpm": random.randint(
            1000,
            2000
        )
    }


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

    if reason_code == 0:

        print(
            "Connected to HiveMQ Cloud!"
        )

    else:

        print(
            f"MQTT connection failed: {reason_code}"
        )


# =========================================================
# CREATE MQTT CLIENT
# =========================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="machine-sensor-simulator"
)


client.username_pw_set(
    USERNAME,
    PASSWORD
)


client.tls_set(
    cert_reqs=ssl.CERT_REQUIRED
)


client.on_connect = on_connect


# =========================================================
# START SIMULATOR
# =========================================================

try:

    print(
        "Starting sensor simulator..."
    )

    client.connect(
        HOST,
        PORT,
        keepalive=60
    )

    client.loop_start()

    print(
        "Sensor simulator started."
    )

    print(
        "Publishing data every 5 seconds."
    )

    print(
        "Press Ctrl+C to stop."
    )


    while True:

        sensor_data = generate_sensor_data()

        payload = json.dumps(
            sensor_data
        )

        result = client.publish(
            TOPIC,
            payload,
            qos=1
        )

        result.wait_for_publish()

        print(
            "Published sensor data:",
            sensor_data
        )

        time.sleep(5)


except KeyboardInterrupt:

    print(
        "\nSensor simulator stopped."
    )


finally:

    client.loop_stop()
    client.disconnect()