


### File: main.py ###
from data_retrieval import get_crypto_data, calculate_metrics
from ml_model import preprocess_data, train_xgboost
import pandas as pd
import joblib

# Parameters
symbol = "BTCUSDT"
interval = "1d"
start_date = "2020-01-01"  # Extended start date for larger dataset
end_date = "2023-01-01"
variable1 = 30  # Look-back period for historical metrics
variable2 = 5  # Look-forward period for future metrics

# Step 1: Get Data
data = get_crypto_data(symbol, interval, start_date, end_date)

# Step 2: Calculate Metrics
data_with_metrics = calculate_metrics(data, variable1, variable2)

# Step 3: Define Features and Targets
features = [
    f'Days_Since_High_Last_{variable1}_Days',
    f'%_Diff_From_High_Last_{variable1}_Days',
    f'Days_Since_Low_Last_{variable1}_Days',
    f'%_Diff_From_Low_Last_{variable1}_Days',
    'RSI', 'EMA_20', 'SMA_50', 'VWAP', 'TWAP'
]
target_high = f'%_Diff_From_High_Next_{variable2}_Days'
target_low = f'%_Diff_From_Low_Next_{variable2}_Days'

# Step 4: Preprocess Data for High and Low Price Prediction (XGBoost)
X_train_high, X_test_high, y_train_high, y_test_high = preprocess_data(data_with_metrics, features, target_high)
X_train_low, X_test_low, y_train_low, y_test_low = preprocess_data(data_with_metrics, features, target_low)

# Step 5: Train XGBoost Model for High Price Prediction
xgb_model_high, xgb_metrics_high = train_xgboost(X_train_high, y_train_high, X_test_high, y_test_high)

# Print XGBoost High Price Model Metrics
print("XGBoost High Price Model Metrics:", xgb_metrics_high)

# Step 6: Train XGBoost Model for Low Price Prediction
xgb_model_low, xgb_metrics_low = train_xgboost(X_train_low, y_train_low, X_test_low, y_test_low)

# Print XGBoost Low Price Model Metrics
print("XGBoost Low Price Model Metrics:", xgb_metrics_low)

# Step 7: Save the Models
joblib.dump(xgb_model_high, 'xgb_model_high.pkl')
joblib.dump(xgb_model_low, 'xgb_model_low.pkl')

# Save Metrics to Excel
metrics_df = pd.DataFrame([
    {"Model": "XGBoost High Price", **xgb_metrics_high},
    {"Model": "XGBoost Low Price", **xgb_metrics_low}
])
with pd.ExcelWriter('model_metrics.xlsx') as writer:
    metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)

print("Process completed successfully.")
