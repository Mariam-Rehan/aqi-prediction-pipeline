from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import hopsworks
import joblib

#generating predictiion data
def generate_prediction_data():
      # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    start_date = datetime.utcnow() 
    end_date = datetime.utcnow() + timedelta(days=4)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
      "latitude": 24.8546842,
      "longitude": 67.0207055,
      "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
      "start_date": start_date.strftime("%Y-%m-%d"),
      "end_date": end_date.strftime("%Y-%m-%d")
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_rain = hourly.Variables(3).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(4).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(8).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
      start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
      end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
      freq = pd.Timedelta(seconds = hourly.Interval()),
      inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["rain"] = hourly_rain
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

    forecast_df = pd.DataFrame(data = hourly_data)
    for lag in range(1, 4):  # Create 3 lags
        forecast_df[f'temp_lag_{lag}'] = forecast_df['temperature_2m'].shift(lag)
    forecast_df['date'] = pd.to_datetime(forecast_df["date"]).dt.tz_localize(None)
    return forecast_df

def get_recent_aqi(feature_group):

    # Simulating the most recent AQI value from the feature store
    latest_data = feature_group.read()
    latest_data = pd.DataFrame(latest_data)
    latest_data.sort_values(by = 'date', inplace=True)
    latest_data = latest_data[['main_aqi','date']]
    return pd.DataFrame(latest_data)

def preprocess_data(forecast_df, feature_group):
    df = feature_group.read()
    df.sort_values(by = 'date', inplace=True)

    recent_aqi = get_recent_aqi(feature_group)
    recent_aqi.sort_values(by="date", inplace=True)

    # Add lag values for the first prediction hour
    forecast_df["aqi_lag_1"] = recent_aqi["main_aqi"].iloc[-1]
    forecast_df["aqi_lag_2"] = recent_aqi["main_aqi"].iloc[-2]
    forecast_df["aqi_lag_3"] = recent_aqi["main_aqi"].iloc[-3]

    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['day_of_year'] = forecast_df['date'].dt.dayofyear
    forecast_df['week_of_year'] = forecast_df['date'].dt.isocalendar().week
    forecast_df['is_weekend'] = forecast_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    forecast_df['hour'] = forecast_df['date'].dt.hour
    
    for lag in range(1, 4):  # Create 3 lags
      forecast_df[f'temp_lag_{lag}'] = df['temperature_2m'].shift(lag)

    forecast_df['aqi_lag_1'].ffill(inplace = True)
    forecast_df['temp_lag_1'].ffill(inplace = True)
    forecast_df['aqi_lag_2'].ffill(inplace = True)
    forecast_df['temp_lag_2'].ffill(inplace = True)
    forecast_df['aqi_lag_3'].ffill(inplace = True)
    forecast_df['temp_lag_3'].ffill(inplace = True)
    forecast_df.drop(columns=['date'], inplace=True)
    return forecast_df

def preprocess_and_predict(forecast_df, model, feature_group):
    import pandas as pd
      # Define the schema for the input data
    input_schema = {
          'temperature_2m': 'float32',
          'relative_humidity_2m': 'float32',
          'precipitation': 'float32',
          'rain': 'float32',
          'pressure_msl': 'float32',
          'surface_pressure': 'float32',
          'wind_speed_10m': 'float32',
          'wind_direction_10m': 'float32',
          'wind_gusts_10m': 'float32',
          'day_of_week': 'int32',
          'month': 'int32',
          'day_of_year': 'int32',
          'week_of_year': 'int64',
          'is_weekend': 'int64',
          'hour': 'int32',
          'aqi_lag_1': 'float64',
          'temp_lag_1': 'float32',
          'aqi_lag_2': 'float64',
          'temp_lag_2': 'float32',
          'aqi_lag_3': 'float64',
          'temp_lag_3': 'float32'
    }
      
      # Create an empty DataFrame with the specified schema
    X_train = pd.DataFrame({
      column: pd.Series(dtype=dtype) for column, dtype in input_schema.items()
    })
      # Get recent AQI data
    recent_aqi = get_recent_aqi(feature_group)
    if len(recent_aqi) < 3:
        raise ValueError("Not enough recent AQI data to populate lag values.")
    
    recent_aqi.sort_values(by="date", inplace=True)
    forecast_df = forecast_df[X_train.columns]  # Ensure all required columns are present

    # Add lag values for the first prediction hour
    forecast_df["aqi_lag_1"] = recent_aqi["main_aqi"].iloc[-1]
    forecast_df["aqi_lag_2"] = recent_aqi["main_aqi"].iloc[-2]
    forecast_df["aqi_lag_3"] = recent_aqi["main_aqi"].iloc[-3]

    # Ensure correct data types
    forecast_df = forecast_df.astype(X_train.dtypes)

    # Predict AQI dynamically
    predicted_aqi = []
    for i in range(len(forecast_df)):
        # Prepare input features for the model
        input_features = forecast_df.iloc[i].copy()
        input_features = input_features.values.reshape(1, -1)  # Ensure proper 2D shape

        # Debug input dimensions
        #print(f"Predicting for row {i} - Input shape: {input_features.shape}, Model expects: {model.n_features_in_}")
        
        # Predict AQI for the current hour
        predicted_value = model.predict(input_features)[0]
        predicted_aqi.append(predicted_value)

        # Update lag values for the next hour
        if i + 1 < len(forecast_df):
            forecast_df.loc[i + 1, "aqi_lag_1"] = predicted_value
            forecast_df.loc[i + 1, "aqi_lag_2"] = forecast_df.loc[i, "aqi_lag_1"]
            forecast_df.loc[i + 1, "aqi_lag_3"] = forecast_df.loc[i, "aqi_lag_2"]

    forecast_df["predicted_aqi"] = predicted_aqi
    forecast_start_date = datetime.utcnow()

    if forecast_start_date.minute >= 30:
        forecast_start_date = forecast_start_date + timedelta(hours=1)

    # Set minutes, seconds, and microseconds to 0
    forecast_start_date = forecast_start_date.replace(minute=0, second=0, microsecond=0)
    # Recreate the date range for the predicted AQI
    date_range = pd.date_range(
        start=forecast_start_date, 
        periods=len(forecast_df),  # Number of rows in the predicted_df
        freq='H'  # Hourly frequency since predictions are hourly
    )

    # Add the date column back to the DataFrame
    forecast_df["date"] = date_range
    forecast_df["predicted_aqi"] = forecast_df["predicted_aqi"].round()

    # If AQI levels should stay within the range [1, 5], ensure the values are clipped
    forecast_df["predicted_aqi"] = forecast_df["predicted_aqi"].clip(1, 5)
    return forecast_df

def get_model():
    import hopsworks
    import xgboost as xgb  # Import the library to load the model
	
    # Log in to Hopsworks
    project = hopsworks.login()
    # Get Hopsworks Model Registry
    mr = project.get_model_registry() 
    # Fetch the model metadata
    model_meta = mr.get_model("aqi_xgboost_model", version=2)
    saved_model_dir = model_meta.download()

    pipeline = joblib.load(saved_model_dir + "/xgboost_pipeline.pkl")
    return pipeline

def get_feature_store():
      project = hopsworks.login(api_key_value = "fGaRronKJ6ZMI6K0.4iXEy9yDd6VkxOypTtfseQQ1Ip3a9sREDzgx6Qnezj50mhqfD7DrzfBlpQPMFAuM")      
      fs = project.get_feature_store()
      feature_group = fs.get_or_create_feature_group(
          name="aqi_weather_features",
          version=2
      )
      return feature_group
