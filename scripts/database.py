import sqlite3
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "machine_health.db"


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL NOT NULL,
            vibration REAL NOT NULL,
            pressure REAL NOT NULL,
            current REAL NOT NULL,
            rpm REAL NOT NULL,
            predicted_status TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print("Database and table created successfully!")
    print(f"Database location: {DB_PATH}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    create_database()