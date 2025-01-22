import streamlit as st
import hopsworks
import pandas as pd
import joblib
import datetime

# Connect to Hopsworks
st.title("Air Quality Dashboard")
st.write("""
This dashboard displays the current Air Quality Index (AQI) and predictions for the next 3 days.
""")

api_key = st.secrets["hopsworks"]["api_key"]
project = hopsworks.login(api_key_value=api_key)

# Get Feature Store
fs = project.get_feature_store()

# Load the current AQI data from the feature store
try:
    feature_view = fs.get_feature_view(name="aqi_features", version=1)  # Replace with your feature view name/version
    current_data = feature_view.get_batch_data()

    # Get the latest data
    latest_data = current_data.sort_values(by="date").iloc[-1]
    st.metric(label="Current AQI", value=f"{latest_data['main_aqi']:.2f}")
    
    # Highlight AQI levels based on the 1-5 range
    if latest_data["main_aqi"] >= 4:
        st.write("🚨 **High AQI**: Avoid outdoor activities!")
    elif latest_data["main_aqi"] == 3:
        st.write("⚠️ **Moderate AQI**: Sensitive groups take precautions.")
    else:
        st.write("✅ **Good AQI**: Air quality is perfect!")

except Exception as e:
    st.error("Error loading current AQI data. Check your feature store setup.")
    st.write(e)

# Load the model from the Hopsworks Model Registry
try:
    mr = project.get_model_registry()
    model = mr.get_model("air_quality_xgboost_model", version=1)  # Replace with your model name/version
    model_dir = model.download()
    xgb_model = joblib.load(f"{model_dir}/model.pkl")  # Adjust path if needed

    # Generate predictions for the next 3 days
    today = datetime.datetime.now()
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, 4)]
    
    # Fetch the most recent data from the feature store to use as input
    latest_features = pd.DataFrame([latest_data])  # Use your fetched `latest_data`
    
    # Replicate the latest features for the next 3 days
    input_data = pd.concat([latest_features] * 3, ignore_index=True)
    
    # If you need to adjust specific features for each day (e.g., incrementing dates)
    input_data["day_of_year"] = [(latest_data["day_of_year"] + i) % 365 for i in range(1, 4)]
    input_data["day_of_week"] = [(latest_data["day_of_week"] + i) % 7 for i in range(1, 4)]
    input_data["is_weekend"] = input_data["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)


    predictions = xgb_model.predict(input_data)

    # Display predictions
    st.subheader("Predicted AQI for the Next 3 Days")
    for date, pred in zip(future_dates, predictions):
        aqi_level = "High" if pred > 150 else "Moderate" if pred > 100 else "Good"
        st.write(f"{date.strftime('%Y-%m-%d')}: {pred:.2f} (Level: {aqi_level})")

except Exception as e:
    st.error("Error loading model or generating predictions. Check your model setup.")
    st.write(e)

# Footer
st.write("Data provided by OpenMeteo and OpenWeather APIs. Powered by Hopsworks.")

