from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class LinearRegressionModel:
    def __init__(self) -> None:
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained: bool = False
        self.feature_names: Optional[List[str]] = None
        self.metrics: Dict[str, float] = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
        """Train the linear regression model"""
        try:
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            self.feature_names = feature_names
            
            # Calculate training metrics
            train_pred = self.predict(X_train)
            self.metrics['train_mse'] = mean_squared_error(y_train, train_pred)
            self.metrics['train_rmse'] = np.sqrt(self.metrics['train_mse'])
            self.metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
            self.metrics['train_r2'] = r2_score(y_train, train_pred)
            
            print(f"Model trained successfully!")
            print(f"Training RMSE: {self.metrics['train_rmse']:.4f}")
            print(f"Training R²: {self.metrics['train_r2']:.4f}")
            return True                        
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        metrics: Dict[str, float] = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        } 
        
        self.metrics.update({f'test_{k}': v for k, v in metrics.items()})
        
        return metrics, predictions

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance (coefficients)"""
        if not self.is_trained:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.coef_))],
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance

    def predict_next_days(self, last_features: np.ndarray, days: int = 5) -> np.ndarray:
        """Predict next N days (simple approach)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_features = last_features.copy()
        
        for _ in range(days):
            pred = self.predict(current_features.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Simple feature update (shift lag features)
            # This is a simplified approach - in practice, you'd need more sophisticated feature engineering
            if len(current_features) > 5:  # Assuming we have lag features
                current_features[1:5] = current_features[0:4]  # Shift lag features
                current_features[0] = pred  # New price becomes lag_1
        
        return np.array(predictions)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """Get model summary"""
        if not self.is_trained:
            return "Model not trained yet"
        
        summary = f"""
            Linear Regression Model Summary:
            ================================
            Features: {len(self.model.coef_)}
            Intercept: {self.model.intercept_:.4f}
            Training R²: {self.metrics.get('train_r2', 'N/A'):.4f}
            Training RMSE: {self.metrics.get('train_rmse', 'N/A'):.4f}
            Test R²: {self.metrics.get('test_r2', 'N/A'):.4f}
            Test RMSE: {self.metrics.get('test_rmse', 'N/A'):.4f}
            """
        return summary