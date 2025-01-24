# Air Quality Prediction using XGBoost and Hopsworks

## Overview
This project leverages machine learning to predict the Air Quality Index (AQI) for Karachi over the next three days. By integrating real-time data from weather APIs, Hopsworks as the feature store, and a Streamlit dashboard for visualization, this project offers a seamless pipeline for monitoring and forecasting air quality levels.

---

## Features
- **Real-Time Data Ingestion:** Hourly updates using Open Meteo and Open Weather APIs.
- **Feature Store Integration:** Efficient data storage and retrieval via Hopsworks.
- **XGBoost Model:** A robust machine learning model trained to predict AQI.
- **Automated Pipelines:** Fully automated feature engineering, model training, and deployment pipelines.
- **Interactive Dashboard:** A user-friendly Streamlit app displaying AQI trends and predictions.

---

##Streamlit App

The Streamlit app provides an interactive way to view AQI predictions.

**Live Demo**

Access the app on the cloud: https://aqi-prediction-mariam.streamlit.app/

---

## Project Architecture
1. **Data Ingestion:**
   - Fetches hourly weather and AQI data from external APIs.
   - Stores the processed data in the Hopsworks feature store.

2. **Feature Engineering:**
   - Constructs lagged features and time-based variables for better prediction.

3. **Model Training:**
   - Daily training of the XGBoost model using the latest features from Hopsworks.
   - Registers the trained model in the Hopsworks Model Registry.

4. **Prediction Pipeline:**
   - Retrieves the latest model and predicts AQI levels for the next three days.
   - Outputs predictions to the Streamlit dashboard.

5. **Visualization:**
   - Displays the current AQI, 3-day predictions, and indicators for air quality levels on the dashboard.

---

## Installation

### Prerequisites
- Python 3.8+
- Hopsworks account and API key
- Dependencies listed in `requirements.txt`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/air-quality-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd air-quality-prediction
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the environment variables:
   - `HOPSWORKS_API_KEY`: Your Hopsworks API key.
   - API keys for Open Meteo and Open Weather.

---

## Usage

### Running the Pipelines
1. **Feature Pipeline:**
   ```bash
   python scripts/feature_pipeline.py
   ```

2. **Training Pipeline:**
   ```bash
   python scripts/train_model.py
   ```

### Starting the Dashboard
Run the Streamlit app:
```bash
streamlit run app/app.py
```

---


## Key Dependencies
- **Hopsworks:** Feature store and model registry.
- **XGBoost:** Machine learning model for AQI prediction.
- **Streamlit:** Interactive dashboard visualization.
- **Pandas:** Data manipulation.
- **Requests:** API integration.

---

## Results
The model predicts AQI levels with indicators for:
- **Good:** <= 2
- **Moderate:** 3
- **Unhealthy:** 4-5

This provides actionable insights for users to plan activities and monitor air quality trends effectively.

---

## Challenges
- Managing real-time API configurations.
- Ensuring data consistency between training and inference.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

