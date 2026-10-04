# AI-Based Industrial Equipment Health Scoring

## 1. Project Overview

This project is an AI-based industrial equipment health monitoring
prototype built using Python, Machine Learning, MQTT, SQLite and
Streamlit.

The system receives simulated industrial sensor readings and predicts
the machine status using a Random Forest classification model.

The monitored parameters are:

- Temperature
- Vibration
- Pressure
- Current
- RPM

The predicted machine conditions are:

- Healthy
- Warning
- Critical

---

## 2. Problem Statement

Industrial machines continuously generate sensor data such as
temperature, vibration, pressure, current and RPM.

Monitoring these parameters manually can make it difficult to identify
potential abnormal conditions quickly.

This project demonstrates a prototype system that collects sensor
readings through MQTT, uses Machine Learning to classify machine
condition, stores the results in SQLite and displays them through a
Streamlit dashboard.

---

## 3. Objectives

The main objectives of this project are:

1. Generate simulated industrial sensor data.
2. Transmit sensor data using MQTT.
3. Apply a Machine Learning model to classify machine condition.
4. Store sensor readings and predictions in SQLite.
5. Display live machine information through a Streamlit dashboard.
6. Visualize sensor trends and machine status distribution.
7. Evaluate the Machine Learning model using standard metrics.

---

## 4. System Architecture

```text
Sensor Simulator
       |
       v
HiveMQ Cloud (MQTT)
       |
       v
MQTT Receiver
       |
       v
Random Forest Model
       |
       v
SQLite Database
       |
       v
Streamlit Dashboard