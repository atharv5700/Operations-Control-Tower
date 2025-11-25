import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def naive_forecast(train_data, horizon):
    """
    Naive forecast: returns the last observed value.
    """
    last_value = train_data.iloc[-1]
    return np.full(horizon, last_value)

def moving_average_forecast(train_data, horizon, window=7):
    """
    Moving Average forecast.
    """
    ma_value = train_data.rolling(window=window).mean().iloc[-1]
    return np.full(horizon, ma_value)

def exponential_smoothing_forecast(train_data, horizon):
    """
    Exponential Smoothing (Holt-Winters).
    """
    try:
        model = ExponentialSmoothing(train_data, seasonal_periods=7, trend='add', seasonal='add', initialization_method="estimated").fit()
        return model.forecast(horizon)
    except:
        # Fallback if ES fails (e.g. too little data)
        return moving_average_forecast(train_data, horizon)

def arima_forecast(train_data, horizon):
    """
    ARIMA forecast (Auto-regressive Integrated Moving Average).
    Simple (1,1,1) order for demonstration speed.
    """
    try:
        model = ARIMA(train_data, order=(1, 1, 1)).fit()
        return model.forecast(steps=horizon)
    except:
        return moving_average_forecast(train_data, horizon)

def prophet_forecast(df_train, horizon):
    """
    Prophet forecast. Expects dataframe with 'ds' and 'y'.
    """
    if not PROPHET_AVAILABLE:
        return None
    
    try:
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        return forecast['yhat'].tail(horizon).values
    except:
        return None

def evaluate_metrics(y_true, y_pred):
    """
    Calculate MAPE, RMSE, Bias.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero for MAPE
    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'Bias': bias
    }
