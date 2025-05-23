# -*- coding: utf-8 -*-

import hopsworks

project = hopsworks.login(api_key_value = "HOPSWORKS_API_KEY")
fs = project.get_feature_store()

from sklearn.ensemble import RandomForestRegressor

# Get or create the 'transactions' feature group
air_quality_fg = fs.get_or_create_feature_group(
    name="aqi_weather_features",
    version=2
)

selected_features = air_quality_fg.select_all()

selected_features.show(5)

feature_view = fs.get_or_create_feature_view(
    name='aqi_fv',
    version=1,
    query=selected_features,
    labels=["main_aqi"]
)

feature_group = fs.get_or_create_feature_group(
    name="aqi_weather_features",
    version=2
)

df = feature_group.read()
df.sort_values(by = 'date', inplace=True)
df.head()

TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_size=TEST_SIZE,
)

# Sort the training features DataFrame 'X_train' based on the 'datetime' column
X_train = X_train.sort_values("index")

# Reindex the target variable 'y_train' to match the sorted order of 'X_train' index
y_train = y_train.reindex(X_train.index)

X_train

# Sort the test features DataFrame 'X_test' based on the 'datetime' column
X_test = X_test.sort_values("index")

# Reindex the target variable 'y_test' to match the sorted order of 'X_test' index
y_test = y_test.reindex(X_test.index)

X_test

# Drop the 'datetime' column from the training features DataFrame 'X_train'
X_train.drop(["date", 'index'], axis=1, inplace=True)

# Drop the 'datetime' column from the test features DataFrame 'X_test'
X_test.drop(["date", 'index'], axis=1, inplace=True)

X_train.columns

y_train.value_counts(normalize=True)

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Initialize XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,   # L1 regularization (try 1, 10, 100)
    reg_lambda=1,  # L2 regularization (try 1, 10, 100)
    random_state=42
)

# Fit the model
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)

# Calculate R² and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost R² Score: {r2}")
print(f"XGBoost Mean Squared Error: {mse}")

# Cross-validation for robustness
# Ensure 'scoring' is a valid metric for XGBRegressor
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}") # Output as positive MSE

mr = project.get_model_registry()

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

model_schema.to_dict()

import os
import joblib

model_dir="aqi_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

joblib.dump(xgb_model, model_dir + '/xgboost_pipeline.pkl')

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# calculate RMSE using sklearn
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

# calculate R squared using sklearn
r2 = r2_score(y_test, y_pred)
print("R squared:", r2)

aq_model = mr.python.create_model(
    name="aqi_xgboost_model",
    metrics={
        "RMSE": rmse,
        "MSE": mse,
        "R squared": r2
    },
    model_schema=model_schema,
    input_example=X_test.sample().values,
    description="AQI predictor.")

aq_model.save(model_dir)
