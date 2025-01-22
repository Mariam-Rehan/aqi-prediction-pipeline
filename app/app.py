import hopsworks
import streamlit as st
import altair as alt
import pandas as pd
from urllib.error import URLError

# Cache data fetching to optimize app performance
@st.cache_data
def get_data():
    # Login to Hopsworks
    project = hopsworks.login(
        project="aqi_prediction_mariam",  # Replace with your Hopsworks project name
        api_key_value="fGaRronKJ6ZMI6K0.4iXEy9yDd6VkxOypTtfseQQ1Ip3a9sREDzgx6Qnezj50mhqfD7DrzfBlpQPMFAuM",  # Fetch API key from Streamlit secrets
    )
    fs = project.get_feature_store()

    # Get the Feature Group for your AQI predictions
    fg = fs.get_feature_group("aqi_features")  # Replace with your feature group name
    return fg.read(online=True)

try:
    # Fetch data from Hopsworks
    df = get_data()

    # Streamlit app interface
    st.write("## Air Quality Index (AQI) Predictions")
    st.write("")
    
    # Display current AQI and predictions for the next 3 days
    current_aqi = df.iloc[-4:]  # Assuming the last 4 rows contain the latest predictions
    st.write("### Current AQI:")
    st.metric(label="AQI Level", value=current_aqi.iloc[0]["main_aqi"], delta="")

    st.write("### Predicted AQI for the Next 3 Days:")
    st.bar_chart(data=current_aqi.set_index("date"), y="main_aqi", use_container_width=True)

except URLError as e:
    # Handle connection issues
    st.error(
        f"""
        **This app requires internet access.**
        Connection error: {e.reason}
        """
    )
