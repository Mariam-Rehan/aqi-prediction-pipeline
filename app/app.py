import hopsworks
import streamlit as st
import altair as alt
import pandas as pd
from urllib.error import URLError

# Cache data fetching to optimize app performance
# Streamlit app interface
st.write("## Air Quality Index (AQI) Predictions")
st.write("")

@st.cache_data()
def get_batch_data_from_fs(td_version):
    st.write(f"Retrieving the Batch data since {date_threshold}")
    feature_view.init_batch_scoring(training_dataset_version=td_version)

    batch_data = feature_view.get_batch_data()
    return batch_data

@st.cache_data()
def download_model(name="air_quality_xgboost_model", version=1):
    mr = project.get_model_registry()
    retrieved_model = mr.get_model(
        name="air_quality_xgboost_model",
        version=1
    )
    saved_model_dir = retrieved_model.download()
    return saved_model_dir

try:
    project = hopsworks.login(
        project="aqi_prediction_mariam",  # Replace with your Hopsworks project name
        api_key_value="fGaRronKJ6ZMI6K0.4iXEy9yDd6VkxOypTtfseQQ1Ip3a9sREDzgx6Qnezj50mhqfD7DrzfBlpQPMFAuM",  # Fetch API key from Streamlit secrets
    )
    fs = project.get_feature_store()
    st.write("✅ Logged in successfully!")

    st.write("Getting the Feature View...")
    feature_view = fs.get_feature_view(
        name = 'aqi_feature_view',
        version = 1
    )
    st.write("✅ Success!")

    print_fancy_header('\n☁️ Retriving batch data from Feature Store...')
    batch_data = get_batch_data_from_fs(td_version=1)

    st.write("Batch data:")
    st.write(batch_data.sample(5))

    saved_model_dir = download_model(
        name="air_quality_xgboost_model",
        version=1
    )

    pipeline = joblib.load(saved_model_dir + "/xgboost_pipeline.pkl")
    st.write("\n")
    st.write("✅ Model was downloaded and cached.")


    
    # Display current AQI and predictions for the next 3 days
    #current_aqi = df.iloc[-4:]  # Assuming the last 4 rows contain the latest predictions
    #st.write("### Current AQI:")
    #st.metric(label="AQI Level", value=current_aqi.iloc[0]["main_aqi"], delta="")

    #st.write("### Predicted AQI for the Next 3 Days:")
    #st.bar_chart(data=current_aqi.set_index("date"), y="main_aqi", use_container_width=True)

except URLError as e:
    # Handle connection issues
    st.error(
        f"""
        **This app requires internet access.**
        Connection error: {e.reason}
        """
    )
