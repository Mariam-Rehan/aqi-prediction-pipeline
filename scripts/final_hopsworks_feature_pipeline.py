# -*- coding: utf-8 -*-

from datetime import timedelta, datetime
import requests
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import hopsworks

def feature_pipeline():

    def fetch_aqi_data():
      today = datetime.utcnow()
      two_years_ago = today - timedelta(days=2*365)
      current_unix_time = int(today.timestamp())

      unix_start = int(two_years_ago.timestamp())  # Convert to UNIX timestamp

      url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8546842&lon=67.0207055&start={unix_start}&end={current_unix_time}&appid=91c226421864cfa90475fb99cdad2ffe"
      response = requests.get(url)
      raw = response.json()

      aqi_df = pd.json_normalize(raw["list"])

      aqi_df['dt'] = pd.to_datetime(aqi_df['dt'], unit='s')
      aqi_df.set_index('dt', inplace=True)
      aqi_df.index = aqi_df.index.tz_localize(None)
      return pd.DataFrame(aqi_df)

    def fetch_weather_data():
          # Setup the Open-Meteo API client with cache and retry on error
          cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
          retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
          openmeteo = openmeteo_requests.Client(session = retry_session)

          start_date = datetime.utcnow() - timedelta(days=2*365)
          end_date = datetime.utcnow() - timedelta(days=1)

          # Make sure all required weather variables are listed here
          # The order of variables in hourly or daily is important to assign them correctly below
          url = "https://archive-api.open-meteo.com/v1/archive"
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

          weather_df = pd.DataFrame(data = hourly_data)
          weather_df['date'] = pd.to_datetime(weather_df["date"]).dt.tz_localize(None)

          return pd.DataFrame(weather_df)

    def fetch_remaining_weather_data():
            # Setup the Open-Meteo API client with cache and retry on error
          cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
          retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
          openmeteo = openmeteo_requests.Client(session = retry_session)

          start_date = datetime.utcnow() - timedelta(days=6)
          end_date = datetime.utcnow() - timedelta(days=1)

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
          remaining_df = pd.DataFrame(data = hourly_data)
          remaining_df['date'] = pd.to_datetime(remaining_df["date"]).dt.tz_localize(None)

          return pd.DataFrame(remaining_df)

    def preprocess_data(aqi_df, weather_df, remaining_df):
          print("Preprocessing Data...")
          print(f"AQI DataFrame Shape: {aqi_df.shape}")
          print(f"Weather DataFrame Shape: {weather_df.shape}")
          print(f"Remaining Weather DataFrame Shape: {remaining_df.shape}")

          # Concatenate weather data
          merged_weather_df = pd.concat([weather_df, remaining_df], axis=0)
          print(f"Merged Weather DataFrame Shape: {merged_weather_df.shape}")

          # Perform the merge using "dt" and "date"
          final_df = pd.merge(aqi_df, merged_weather_df, left_on="dt", right_on="date", how="inner")
          print(f"Final Merged DataFrame Shape: {final_df.shape}")

          # Check NaN counts in the date column
          print(f"Number of NaNs in 'date': {final_df['date'].isna().sum()}")

          final_df.columns = final_df.columns.str.replace(r"\.", "_", regex=True)
          final_df.columns = final_df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
          final_df.columns = final_df.columns.str.lower()

          final_df['date'] = pd.to_datetime(final_df['date'])
          final_df['day_of_week'] = final_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
          final_df['month'] = final_df['date'].dt.month
          final_df['day_of_year'] = final_df['date'].dt.dayofyear
          final_df['week_of_year'] = final_df['date'].dt.isocalendar().week
          final_df['is_weekend'] = final_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
          final_df['hour'] = final_df['date'].dt.hour

          for lag in range(1, 4):  # Create 3 lags
              final_df[f'aqi_lag_{lag}'] = final_df['main_aqi'].shift(lag)
              final_df[f'temp_lag_{lag}'] = final_df['temperature_2m'].shift(lag)

          final_df['aqi_lag_1'].bfill(inplace = True)
          final_df['temp_lag_1'].bfill(inplace = True)
          final_df['aqi_lag_2'].bfill(inplace = True)
          final_df['temp_lag_2'].bfill(inplace = True)
          final_df['aqi_lag_3'].bfill(inplace = True)
          final_df['temp_lag_3'].bfill(inplace = True)

          final_df.drop(columns = ['components_co', 'components_no', 'components_no2','components_o3', 'components_so2', 'components_pm2_5','components_pm10', 'components_nh3'], axis = 1, inplace = True)

          # Fixing duplicate issues while prioritizing rows with fewer NaNs
          df_cleaned = (
              final_df.sort_values("date", kind="mergesort")
              .assign(missing_count=final_df.isna().sum(axis=1))
              .sort_values(by=["date", "missing_count"])
              .drop_duplicates(subset="date", keep="first")
              .drop(columns=["missing_count"])
          )

          df_cleaned.reset_index(drop=True, inplace=True)
          print(f"Cleaned DataFrame Shape: {df_cleaned.shape}")
          df_cleaned.dropna(inplace = True)
          df_cleaned["index"] = df_cleaned.index
          return df_cleaned

    aqi_df = fetch_aqi_data()
    weather_df = fetch_weather_data()
    remaining_df = fetch_remaining_weather_data()
    final_df = preprocess_data(aqi_df, weather_df, remaining_df)
    return final_df

#Insert into Feature Store on Hopsworks
project = hopsworks.login(api_key_value = "HOPSWORKS_API_KEY")
fs = project.get_feature_store()

air_quality_fg = fs.get_or_create_feature_group(
    name="aqi_weather_features",
    version=2,
)

df = feature_pipeline()
air_quality_fg.insert(df)
