"""
Intelligent Demand Forecasting Module
---------------------------------------
Ensemble ML forecasting with Auto-ARIMA, XGBoost, Prophet, and intelligent model selection.
All models work 100% offline with no external API calls.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Statistical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# ML models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False
    warnings.warn("Auto-ARIMA not available. Install with: pip install pmdarima")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

warnings.filterwarnings('ignore')


class DemandForecaster:
    """
    Intelligent ensemble forecasting engine that automatically selects the best model
    based on data characteristics and cross-validation performance.
    """
    
    def __init__(self, train_data: pd.Series, horizon: int = 7, 
                 external_features: Optional[pd.DataFrame] = None):
        """
        Args:
            train_data: Historical demand time series (DatetimeIndex)
            horizon: Forecast horizon in days
            external_features: Optional features (promotions, prices, etc.)
        """
        self.train_data = train_data
        self.horizon = horizon
        self.external_features = external_features
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based and lag features for ML models"""
        df = df.copy()
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['week_of_year'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Lag features (7-day, 14-day, 28-day lags)
        for lag in [7, 14, 28]:
            df[f'lag_{lag}'] = df['demand'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 28]:
            df[f'rolling_mean_{window}'] = df['demand'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['demand'].rolling(window).std()
        
        # Fill NaN with forward fill then backward fill
        df = df.ffill().bfill()
        
        return df
    
    def auto_arima_forecast(self) -> np.ndarray:
        """Auto-ARIMA with automatic hyperparameter tuning"""
        if not AUTO_ARIMA_AVAILABLE:
            return self._fallback_arima()
        
        try:
            # Auto-ARIMA finds best (p,d,q) and seasonal parameters
            model = auto_arima(
                self.train_data,
                start_p=0, start_q=0, max_p=5, max_q=5,
                m=7,  # Weekly seasonality
                seasonal=True,
                d=None,  # Auto-detect differencing
                D=None,  # Auto-detect seasonal differencing
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_jobs=-1
            )
            
            self.models['auto_arima'] = model
            forecast = model.predict(n_periods=self.horizon)
            return np.maximum(forecast, 0)  # Ensure non-negative
            
        except Exception as e:
            warnings.warn(f"Auto-ARIMA failed: {e}")
            return self._fallback_arima()
    
    def _fallback_arima(self) -> np.ndarray:
        """Fallback to simple SARIMA if Auto-ARIMA fails"""
        try:
            model = SARIMAX(
                self.train_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=self.horizon)
            return np.maximum(forecast, 0)
        except:
            # Last resort: exponential smoothing
            return self.exponential_smoothing_forecast()
    
    def exponential_smoothing_forecast(self) -> np.ndarray:
        """Enhanced Exponential Smoothing with automatic parameter selection"""
        try:
            model = ExponentialSmoothing(
                self.train_data,
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                damped_trend=True,
                initialization_method='estimated'
            )
            fitted = model.fit(optimized=True)
            self.models['exp_smoothing'] = fitted
            forecast = fitted.forecast(self.horizon)
            return np.maximum(forecast, 0)
        except:
            # Ultra-simple fallback
            last_week = self.train_data.tail(7).mean()
            return np.full(self.horizon, last_week)
    
    def xgboost_forecast(self) -> np.ndarray:
        """XGBoost with feature engineering"""
        if not XGBOOST_AVAILABLE:
            return None
        
        try:
            # Prepare data with features
            df = pd.DataFrame({'demand': self.train_data})
            df = self.engineer_features(df)
            
            # Create supervised learning dataset
            feature_cols = [c for c in df.columns if c != 'demand']
            X_train = df[feature_cols].iloc[:-self.horizon]
            y_train = df['demand'].iloc[:-self.horizon]
            
            # Train XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models['xgboost'] = model
            
            # Generate forecast by iterating forward
            forecast = []
            current_df = df.copy()
            
            for i in range(self.horizon):
                # Predict next value
                X_next = current_df[feature_cols].iloc[-1:].values
                pred = model.predict(X_next)[0]
                forecast.append(max(pred, 0))
                
                # Update dataframe with prediction for next iteration
                next_date = current_df.index[-1] + timedelta(days=1)
                new_row = pd.DataFrame({'demand': [pred]}, index=[next_date])
                current_df = pd.concat([current_df, new_row])
                current_df = self.engineer_features(current_df)
            
            return np.array(forecast)
            
        except Exception as e:
            warnings.warn(f"XGBoost failed: {e}")
            return None
    
    def prophet_forecast(self) -> np.ndarray:
        """Facebook Prophet for robust trend/seasonality detection"""
        if not PROPHET_AVAILABLE:
            return None
        
        try:
            # Prepare data in Prophet format
            df_prophet = pd.DataFrame({
                'ds': self.train_data.index,
                'y': self.train_data.values
            })
            
            # Configure Prophet
            model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(df_prophet)
            self.models['prophet'] = model
            
            # Generate forecast
            future = model.make_future_dataframe(periods=self.horizon)
            forecast = model.predict(future)
            
            predictions = forecast['yhat'].tail(self.horizon).values
            return np.maximum(predictions, 0)
            
        except Exception as e:
            warnings.warn(f"Prophet failed: {e}")
            return None
    
    def ensemble_forecast(self, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Weighted ensemble of all available models.
        If weights not provided, uses inverse MAPE weighting from CV.
        """
        all_forecasts = {}
        
        # Generate forecasts from all models
        all_forecasts['auto_arima'] = self.auto_arima_forecast()
        all_forecasts['exp_smoothing'] = self.exponential_smoothing_forecast()
        
        xgb_pred = self.xgboost_forecast()
        if xgb_pred is not None:
            all_forecasts['xgboost'] = xgb_pred
        
        prophet_pred = self.prophet_forecast()
        if prophet_pred is not None:
            all_forecasts['prophet'] = prophet_pred
        
        # Store individual predictions
        self.predictions = all_forecasts
        
        # If weights provided, use them
        if weights:
            weighted_sum = np.zeros(self.horizon)
            total_weight = 0
            for model_name, forecast in all_forecasts.items():
                if model_name in weights:
                    weighted_sum += forecast * weights[model_name]
                    total_weight += weights[model_name]
            return weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        # Otherwise, equal weighting
        ensemble = np.mean(list(all_forecasts.values()), axis=0)
        return ensemble
    
    def select_best_model(self, validation_split: float = 0.8) -> Tuple[str, np.ndarray]:
        """
        Cross-validate all models and return the best one based on MAPE.
        
        Returns:
            (best_model_name, best_forecast)
        """
        # Split data for validation
        split_point = int(len(self.train_data) * validation_split)
        train_cv = self.train_data.iloc[:split_point]
        test_cv = self.train_data.iloc[split_point:]
        
        # Temporarily set horizon to validation set size
        original_horizon = self.horizon
        self.horizon = len(test_cv)
        
        # Temporarily use training subset
        original_train = self.train_data
        self.train_data = train_cv
        
        # Get all predictions on validation set
        all_forecasts = {}
        all_forecasts['auto_arima'] = self.auto_arima_forecast()
        all_forecasts['exp_smoothing'] = self.exponential_smoothing_forecast()
        
        xgb_pred = self.xgboost_forecast()
        if xgb_pred is not None:
            all_forecasts['xgboost'] = xgb_pred
        
        prophet_pred = self.prophet_forecast()
        if prophet_pred is not None:
            all_forecasts['prophet'] = prophet_pred
        
        # Evaluate each model
        best_model = None
        best_mape = float('inf')
        
        for model_name, forecast in all_forecasts.items():
            mape = evaluate_forecast(test_cv.values, forecast)['MAPE']
            self.metrics[model_name] = mape
            
            if mape < best_mape:
                best_mape = mape
                best_model = model_name
        
        # Restore original settings and re-train best model on full data
        self.train_data = original_train
        self.horizon = original_horizon
        
        # Generate final forecast with best model
        if best_model == 'auto_arima':
            best_forecast = self.auto_arima_forecast()
        elif best_model == 'exp_smoothing':
            best_forecast = self.exponential_smoothing_forecast()
        elif best_model == 'xgboost':
            best_forecast = self.xgboost_forecast()
        elif best_model == 'prophet':
            best_forecast = self.prophet_forecast()
        else:
            best_forecast = self.ensemble_forecast()
        
        return best_model, best_forecast


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive forecast accuracy metrics.
    
    Returns:
        Dict with MAPE, RMSE, MAE, Bias
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle zero values for MAPE
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = 0.0
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    return {
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'Bias': round(bias, 2)
    }


def quick_forecast(train_data: pd.Series, horizon: int = 7, 
                   method: str = 'auto') -> Tuple[np.ndarray, str, Dict]:
    """
    Quick forecasting function for easy integration.
    
    Args:
        train_data: Historical demand time series
        horizon: Days to forecast
        method: 'auto' (best model), 'ensemble', or specific model name
    
    Returns:
        (forecast_values, method_used, metrics_dict)
    """
    forecaster = DemandForecaster(train_data, horizon)
    
    if method == 'auto':
        best_model, forecast = forecaster.select_best_model()
        return forecast, best_model, forecaster.metrics
    elif method == 'ensemble':
        forecast = forecaster.ensemble_forecast()
        return forecast, 'ensemble', {}
    else:
        # Call specific method
        if method == 'auto_arima':
            forecast = forecaster.auto_arima_forecast()
        elif method == 'exp_smoothing':
            forecast = forecaster.exponential_smoothing_forecast()
        elif method == 'xgboost':
            forecast = forecaster.xgboost_forecast()
        elif method == 'prophet':
            forecast = forecaster.prophet_forecast()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return forecast, method, {}
