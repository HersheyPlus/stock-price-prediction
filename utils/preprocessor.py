from typing import Tuple, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



class StockDataPreprocessor:
    def __init__(self) -> None:
        self.scaler = MinMaxScaler()
        self.feature_columns: Optional[List[str]] = None
        self.target_column: str = 'Close'
            
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """Prepare features for ML models"""
        df = data.copy()
        self.feature_columns = target_col
        
        # Remove NaN / NA values
        df = df.dropna()
        
        # Create lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df[target_col].shift(lag)
        
        # Price change features
        df['Price_change'] = df[target_col].pct_change()
        df['Price_change_2d'] = df[target_col].pct_change(periods=2)
        df['Price_change_5d'] = df[target_col].pct_change(periods=5)

        # Volatility features
        df['Volatility_5d'] = df[target_col].rolling(window=5).std()
        df['Volatility_10d'] = df[target_col].rolling(window=10).std()

        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA_5']
        
        # Remove rows with NaN after feature creation
        df = df.dropna()
        
        return df
    
    def split_train_test(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""

        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        return train_data, test_data
    
    def prepare_ml_data(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Prepare data for traditional ML models"""
        df = self.prepare_features(data, target_col)
        
        # Feature columns (exclude target and non-numeric columns)
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y, df.index

    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str = 'Close', time_steps: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        df = data.copy()
        
        # Use only close prices for LSTM
        prices = df[target_col].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions"""
        return self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    def create_sequences(self, data: np.ndarray, time_steps: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)