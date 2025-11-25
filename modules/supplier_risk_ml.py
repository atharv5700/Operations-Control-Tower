"""
Intelligent Supplier Risk ML Model
-----------------------------------
Predictive machine learning classifier for supplier reliability.
Includes SHAP explainability to show why suppliers are risky.
100% offline - all models trained locally, no external APIs.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional

# ML models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available - model explanations will be limited")

warnings.filterwarnings('ignore')


class SupplierRiskPredictor:
    """
    ML-based supplier risk prediction with explainability.
    Predicts: Will this supplier deliver late?
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Args:
            model_type: 'xgboost', 'random_forest', or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.explainer = None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for supplier risk prediction.
        
        Expected columns in df:
            - Supplier_ID
            - Supplier_Lead_Time_Days
            - Date
            - Units_Sold (demand)
            - Stockout_Flag (optional)
        """
        # Group by supplier
        supplier_features = []
        
        for supplier_id in df['Supplier_ID'].unique():
            supplier_df = df[df['Supplier_ID'] == supplier_id].copy()
            
            # Historical statistics
            lt_mean = supplier_df['Supplier_Lead_Time_Days'].mean()
            lt_std = supplier_df['Supplier_Lead_Time_Days'].std()
            lt_max = supplier_df['Supplier_Lead_Time_Days'].max()
            lt_min = supplier_df['Supplier_Lead_Time_Days'].min()
            lt_cv = (lt_std / lt_mean) if lt_mean > 0 else 0  # Coefficient of variation
            
            # Trend analysis (is lead time increasing?)
            if len(supplier_df) > 10:
                recent_lt = supplier_df.tail(10)['Supplier_Lead_Time_Days'].mean()
                older_lt = supplier_df.head(10)['Supplier_Lead_Time_Days'].mean()
                lt_trend = (recent_lt - older_lt) / (older_lt + 1)
            else:
                lt_trend = 0
            
            # Stockout correlation
            if 'Stockout_Flag' in supplier_df.columns:
                stockout_rate = supplier_df['Stockout_Flag'].mean()
            else:
                stockout_rate = 0
            
            # Order frequency and volume
            total_orders = len(supplier_df)
            avg_order_size = supplier_df['Units_Sold'].mean() if 'Units_Sold' in supplier_df.columns else 0
            
            # Seasonality factor (does lead time vary by month?)
            if 'Date' in supplier_df.columns and len(supplier_df) > 30:
                supplier_df['month'] = pd.to_datetime(supplier_df['Date']).dt.month
                monthly_variation = supplier_df.groupby('month')['Supplier_Lead_Time_Days'].std().mean()
            else:
                monthly_variation = 0
            
            # Create feature row
            features = {
                'Supplier_ID': supplier_id,
                'lt_mean': lt_mean,
                'lt_std': lt_std,
                'lt_max': lt_max,
                'lt_min': lt_min,
                'lt_range': lt_max - lt_min,
                'lt_cv': lt_cv,
                'lt_trend': lt_trend,
                'stockout_rate': stockout_rate,
                'total_orders': total_orders,
                'avg_order_size': avg_order_size,
                'monthly_variation': monthly_variation
            }
            
            supplier_features.append(features)
        
        return pd.DataFrame(supplier_features)
    
    def create_labels(self, df: pd.DataFrame, threshold_percentile: float = 75) -> pd.Series:
        """
        Create binary labels: 1 = High Risk, 0 = Low Risk
        
        High risk = Lead time in top 25% (above 75th percentile)
        """
        threshold = df['Supplier_Lead_Time_Days'].quantile(threshold_percentile / 100)
        labels = (df['Supplier_Lead_Time_Days'] > threshold).astype(int)
        
        return labels
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train supplier risk prediction model.
        
        Returns:
            Metrics dict with accuracy, precision, recall, AUC
        """
        # Feature engineering
        features_df = self.engineer_features(df)
        
        # Create labels (per supplier)
        # Calculate risk based on lead time variability
        features_df['is_high_risk'] = (
            (features_df['lt_cv'] > features_df['lt_cv'].median()) |
            (features_df['lt_trend'] > 0.1) |
            (features_df['stockout_rate'] > 0.1)
        ).astype(int)
        
        # Prepare features
        feature_cols = [c for c in features_df.columns if c not in ['Supplier_ID', 'is_high_risk']]
        self.feature_names = feature_cols
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['is_high_risk']
        
        # Check class balance
        if y.sum() == 0 or y.sum() == len(y):
            warnings.warn("All suppliers have same risk level - model may not be useful")
            self.model = None
            return {'error': 'Insufficient diversity in data'}
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        else:  # Default: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 3),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 3),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 3),
            'auc': round(roc_auc_score(y_test, y_proba), 3) if len(np.unique(y_test)) > 1 else 0
        }
        
        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            try:
                if self.model_type == 'xgboost':
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    self.explainer = shap.Explainer(self.model, X_train_scaled)
            except:
                self.explainer = None
        
        return metrics
    
    def predict_risk(self, supplier_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk for new suppliers.
        
        Returns:
            DataFrame with risk_score (0-1) and risk_level (Low/Medium/High)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = supplier_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        risk_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Classify risk level
        risk_level = pd.cut(
            risk_proba,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        results = supplier_features[['Supplier_ID']].copy()
        results['risk_score'] = risk_proba
        results['risk_level'] = risk_level
        results['on_time_probability'] = 1 - risk_proba
        
        return results
    
    def explain_prediction(self, supplier_id: str, supplier_features: pd.DataFrame) -> Dict:
        """
        Explain why a supplier is risky using SHAP values.
        
        Returns:
            Dict with top risk factors
        """
        if self.explainer is None:
            return {'error': 'SHAP not available or model not trained'}
        
        # Get features for this supplier
        supplier_row = supplier_features[supplier_features['Supplier_ID'] == supplier_id]
        
        if supplier_row.empty:
            return {'error': 'Supplier not found'}
        
        X = supplier_row[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_scaled)
            
            # Get feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(shap_values[0]),
                'value': X.iloc[0].values
            }).sort_values('importance', ascending=False)
            
            # Format top factors
            top_factors = []
            for _, row in feature_importance.head(5).iterrows():
                factor_name = row['feature'].replace('_', ' ').title()
                factor_value = row['value']
                top_factors.append(f"{factor_name}: {factor_value:.2f}")
            
            return {
                'supplier_id': supplier_id,
                'top_risk_factors': top_factors
            }
        
        except Exception as e:
            return {'error': f'Explanation failed: {e}'}


def calculate_traditional_reliability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: Traditional supplier scoring if ML model can't be trained.
    
    Returns:
        DataFrame with Supplier_ID and Reliability_Score (0-100)
    """
    supplier_stats = df.groupby('Supplier_ID').agg({
        'Supplier_Lead_Time_Days': ['mean', 'std', 'count'],
        'Stockout_Flag': 'mean' if 'Stockout_Flag' in df.columns else lambda x: 0
    }).reset_index()
    
    supplier_stats.columns = ['Supplier_ID', 'Avg_LT', 'Std_LT', 'Order_Count', 'Stockout_Rate']
    
    # Fill NaN
    supplier_stats['Std_LT'] = supplier_stats['Std_LT'].fillna(0)
    supplier_stats['Stockout_Rate'] = supplier_stats['Stockout_Rate'].fillna(0)
    
    # Normalize metrics
    max_std = supplier_stats['Std_LT'].max()
    if max_std == 0:
        max_std = 1
    
    # Scoring formula (100 = perfect)
    # Penalize: high stockout rate, high lead time variability
    supplier_stats['Reliability_Score'] = 100 - (
        supplier_stats['Stockout_Rate'] * 100 * 0.4 +  # 40% weight
        (supplier_stats['Std_LT'] / max_std) * 100 * 0.3 +  # 30% weight
        (supplier_stats['Avg_LT'] / supplier_stats['Avg_LT'].max()) * 100 * 0.3  # 30% weight
    )
    
    supplier_stats['Reliability_Score'] = supplier_stats['Reliability_Score'].clip(0, 100)
    
    return supplier_stats[['Supplier_ID', 'Reliability_Score', 'Avg_LT', 'Std_LT', 'Stockout_Rate']]
