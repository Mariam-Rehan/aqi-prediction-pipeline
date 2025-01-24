import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import generate_prediction_data, preprocess_data, preprocess_and_predict, get_model
import hopsworks
import os
# Display a title
st.title("Real-time AQI Forecast")

# Log in to Hopsworks using the API key
project = hopsworks.login(project='aqi_prediction_mariam',api_key_value="fGaRronKJ6ZMI6K0.4iXEy9yDd6VkxOypTtfseQQ1Ip3a9sREDzgx6Qnezj50mhqfD7DrzfBlpQPMFAuM")
fs = project.get_feature_store()
feature_group = fs.get_or_create_feature_group(
    name="aqi_weather_features",
    version=2
)
# Function to display the current AQI
current_aqi = get_recent_aqi(feature_group)  # Assuming you have this function
current_aqi_value = current_aqi['main_aqi'].iloc[-1]  # Get the latest AQI value
st.write(f"### Current AQI: {current_aqi_value}")
if current_aqi_value <= 2:
    st.write("Air quality is **Good**. You're all set!")
elif current_aqi_value == 3:
    st.write("Air quality is **Moderate**. It may not be a problem for most people.")
elif current_aqi_value == 4:
    st.write("Air quality is **Unhealthy for Sensitive Groups**. Those with respiratory issues should limit outdoor activities.")
elif current_aqi_value == 5:
    st.write("Air quality is **Unhealthy**. Everyone may begin to experience health effects.")

# Display predicted AQI for the next 3 days
# Forecast data is in prediction_df
st.write("### Predicted AQI for the next 3 days (Hourly)")

# Extract data for the next 3 days
forecast_df = generate_prediction_data()
forecast_df = preprocess_data(forecast_df, feature_group)
xgb_model = get_model()
prediction_df = preprocess_and_predict(forecast_df, xgb_model, feature_group)

prediction_df['date'] = pd.to_datetime(prediction_df['date'])
prediction_df['hour'] = prediction_df['date'].dt.hour
prediction_df['day'] = prediction_df['date'].dt.date

# Plot AQI predictions
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=prediction_df, x="hour", y="predicted_aqi", hue="day", ax=ax, marker="o")

ax.set_title('Hourly AQI Prediction for the Next 3 Days')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Predicted AQI')
st.pyplot(fig)

# Run the app
#if __name__ == '__main__':
# Display current AQI
#   display_current_aqi()

# Display the predicted AQI
#  display_predicted_aqi()
