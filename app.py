import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# Page setup
st.set_page_config(page_title="Temperature Prediction & Automation", layout="centered")
st.title("🌡️ Temperature Prediction & Fan Automation System")
st.write("This app predicts future temperatures using a simple ML model and simulates a fan control system.")

# Function to generate synthetic temperature data
@st.cache_data
def generate_temperature_data(n_days=100):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
    day_nums = np.arange(n_days)
    hours = np.random.randint(0, 24, size=n_days)
    noise = np.random.normal(0, 2, size=n_days)
    temps = 25 + 0.1 * day_nums + 3 * np.sin(2 * np.pi * hours / 24) + noise
    df = pd.DataFrame({
        "Date": dates,
        "Day": day_nums,
        "Hour": hours,
        "Temperature": temps
    })
    return df

# Train linear regression model
def train_model(df):
    X = df[["Day", "Hour"]]
    y = df["Temperature"]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict temperature without warnings
def predict_temperature(model, day, hour):
    input_df = pd.DataFrame([[day, hour]], columns=["Day", "Hour"])
    temp = model.predict(input_df)[0]
    return round(temp, 2)

# Plot temperature data and prediction
def plot_temperature(df, model):
    df_sorted = df.sort_values(by="Day")
    X = df_sorted[["Day", "Hour"]]
    y_true = df_sorted["Temperature"]
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_sorted["Date"], y_true, label="Actual Temperature")
    ax.plot(df_sorted["Date"], y_pred, label="Predicted Temperature", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("📈 Actual vs Predicted Temperature")
    ax.legend()
    st.pyplot(fig)

# UI Section - Data simulation and training
st.subheader("📊 Generate Dataset and Train Model")
n_days = st.slider("Number of past days to simulate", 50, 300, 100, key="days_slider")
df = generate_temperature_data(n_days)
model = train_model(df)

# Visualization
plot_temperature(df, model)

# UI Section - Prediction
st.subheader("🔮 Predict Future Temperature")
col1, col2 = st.columns(2)
with col1:
    future_day = st.number_input("Days into the future (from now)", min_value=1, max_value=365, value=7, key="future_day")
with col2:
    hour = st.slider("Hour of the day", 0, 23, 12, key="hour_slider")

predicted_temp = predict_temperature(model, n_days + future_day, hour)
st.metric(label="Predicted Temperature", value=f"{predicted_temp} °C")

# Automation logic
if predicted_temp > 30:
    st.success("💨 FAN ON (Temperature > 30°C)")
else:
    st.info("❄️ FAN OFF (Temperature ≤ 30°C)")

# Fix Arrow serialization error by converting timestamps to string
df["Date"] = df["Date"].astype(str)

# Dataset statistics
st.subheader("📈 Dataset Statistics")
st.write(df.describe())
