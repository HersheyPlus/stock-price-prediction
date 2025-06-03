from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        self.figsize = figsize
    
    def plot_price_history(self, data: pd.DataFrame, symbol: str = "Stock", columns: List[str] = ['Close']) -> None:
        """Plot historical stock prices"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for col in columns:
            if col in data.columns:
                ax.plot(data.index, data[col], label=col, linewidth=2)
        
        ax.set_title(f'{symbol} Price History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_volume_analysis(self, data: pd.DataFrame, symbol: str = "Stock") -> None:
        """Plot volume analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        
        # Price plot
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=2)
        ax1.set_title(f'{symbol} Price and Volume Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_technical_indicators(self, data: pd.DataFrame, symbol: str = "Stock") -> None:
        """Plot technical indicators"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price with Moving Averages
        ax1.plot(data.index, data['Close'], label='Close', linewidth=2)
        if 'MA_20' in data.columns:
            ax1.plot(data.index, data['MA_20'], label='MA 20', alpha=0.7)
        if 'MA_50' in data.columns:
            ax1.plot(data.index, data['MA_50'], label='MA 50', alpha=0.7)
        ax1.set_title(f'{symbol} Price with Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bollinger Bands
        ax2.plot(data.index, data['Close'], label='Close', linewidth=2)
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            ax2.plot(data.index, data['BB_upper'], label='BB Upper', alpha=0.7)
            ax2.plot(data.index, data['BB_lower'], label='BB Lower', alpha=0.7)
            ax2.fill_between(data.index, data['BB_lower'], data['BB_upper'], alpha=0.1)
        ax2.set_title('Bollinger Bands')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in data.columns:
            ax3.plot(data.index, data['RSI'], label='RSI', color='purple', linewidth=2)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            ax3.set_title('RSI (Relative Strength Index)')
            ax3.set_ylabel('RSI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Volume
        ax4.bar(data.index, data['Volume'], alpha=0.7, color='orange')
        ax4.set_title('Trading Volume')
        ax4.set_ylabel('Volume')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_comparison(self, dates: pd.DatetimeIndex, actual: np.ndarray, 
                                  lr_pred: np.ndarray, lstm_pred: np.ndarray, symbol: str = "Stock") -> None:
        """Compare predictions from different models"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, actual, label='Actual', linewidth=2, color='black')
        ax.plot(dates, lr_pred, label='Linear Regression', linewidth=2, alpha=0.8)
        ax.plot(dates, lstm_pred, label='LSTM', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{symbol} - Model Predictions Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_accuracy(self, actual: np.ndarray, predicted: np.ndarray, model_name: str) -> None:
        """Plot prediction accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Actual vs Predicted scatter
        ax1.scatter(actual, predicted, alpha=0.6)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title(f'{model_name} - Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = actual - predicted
        ax2.scatter(predicted, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Price')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name} - Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_future_predictions(self, historical_data: pd.DataFrame, future_dates: pd.DatetimeIndex, 
                              future_predictions: np.ndarray, model_name: str, symbol: str = "Stock", 
                              days_history: int = 30) -> None:
        """Plot future predictions with historical context"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Historical data (last N days)
        recent_data = historical_data.tail(days_history)
        ax.plot(recent_data.index, recent_data['Close'], 
                label='Historical', linewidth=2, color='blue')
        
        # Future predictions
        ax.plot(future_dates, future_predictions, 
                label=f'{model_name} Predictions', linewidth=2, 
                color='red', marker='o', markersize=4)
        
        # Connect last historical point to first prediction
        if len(recent_data) > 0 and len(future_predictions) > 0:
            ax.plot([recent_data.index[-1], future_dates[0]], 
                   [recent_data['Close'].iloc[-1], future_predictions[0]], 
                   'r--', alpha=0.5)
        
        ax.set_title(f'{symbol} - {model_name} Future Predictions', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_model_metrics_comparison(self, lr_metrics: Dict[str, float], lstm_metrics: Dict[str, float]) -> None:
        """Compare model metrics"""
        metrics = ['rmse', 'mae', 'mape']
        
        lr_values = [lr_metrics.get(f'test_{m}', 0) for m in metrics]
        lstm_values = [lstm_metrics.get(f'test_{m}', 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, lr_values, width, label='Linear Regression', alpha=0.8)
        bars2 = ax.bar(x + width/2, lstm_values, width, label='LSTM', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def autolabel(rects) -> None:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Feature Correlation Matrix") -> None:
        """Plot correlation matrix"""
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   center=0, square=True, ax=ax, cmap='coolwarm')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_price_distribution(self, data: pd.DataFrame, symbol: str = "Stock") -> None:
        """Plot price distribution analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price histogram
        ax1.hist(data['Close'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title(f'{symbol} Price Distribution')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Daily returns histogram
        returns = data['Close'].pct_change().dropna()
        ax2.hist(returns, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Price over time with volatility
        ax3.plot(data.index, data['Close'], linewidth=1)
        rolling_std = data['Close'].rolling(window=20).std()
        ax3.fill_between(data.index, 
                        data['Close'] - rolling_std, 
                        data['Close'] + rolling_std, 
                        alpha=0.2)
        ax3.set_title('Price with Volatility Bands')
        ax3.set_ylabel('Price ($)')
        ax3.grid(True, alpha=0.3)
        
        # Box plot by month
        data_copy = data.copy()
        data_copy['Month'] = data_copy.index.month
        monthly_data = [data_copy[data_copy['Month'] == m]['Close'].values 
                       for m in range(1, 13)]
        ax4.boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax4.set_title('Monthly Price Distribution')
        ax4.set_ylabel('Price ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_loss(self, history: Dict[str, List[float]], title: str = "Training History") -> None:
        """Plot training loss and metrics"""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics plot (MAE if available)
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Training MAE')
            if 'val_mae' in history:
                axes[1].plot(history['val_mae'], label='Validation MAE')
            axes[1].set_title('Model MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15) -> None:
        """Plot feature importance for linear regression"""
        if importance_df is None or importance_df.empty:
            print("No feature importance data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['abs_coefficient'], alpha=0.7)
        
        # Color bars based on positive/negative coefficients
        for i, (bar, coef) in enumerate(zip(bars, top_features['coefficient'])):
            if coef >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title(f'Top {top_n} Feature Importance (Linear Regression)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, coef) in enumerate(zip(bars, top_features['abs_coefficient'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{coef:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_price_prediction_timeline(self, data: pd.DataFrame, train_pred: np.ndarray, 
                                     test_pred: np.ndarray, train_size: int, 
                                     model_name: str, symbol: str = "Stock") -> None:
        """Plot price predictions over time"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Actual prices
        ax.plot(data.index, data['Close'], label='Actual Price', linewidth=2, color='black')
        
        # Training predictions
        train_dates = data.index[:train_size]
        if len(train_pred) == len(train_dates):
            ax.plot(train_dates, train_pred, label='Training Predictions', 
                   linewidth=1, alpha=0.7, color='blue')
        
        # Test predictions
        test_dates = data.index[train_size:train_size+len(test_pred)]
        ax.plot(test_dates, test_pred, label='Test Predictions', 
               linewidth=2, alpha=0.8, color='red')
        
        # Add vertical line to separate train/test
        ax.axvline(x=data.index[train_size], color='gray', linestyle='--', alpha=0.7, 
                  label='Train/Test Split')
        
        ax.set_title(f'{symbol} - {model_name} Predictions Timeline', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_error_analysis(self, actual: np.ndarray, predicted: np.ndarray, 
                           model_name: str) -> None:
        """Plot comprehensive error analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual vs Predicted
        ax1.scatter(actual, predicted, alpha=0.6)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = actual - predicted
        ax2.scatter(predicted, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Price')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residuals Distribution')
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.grid(True, alpha=0.3)
        
        # Error over time
        errors = np.abs(residuals)
        ax4.plot(range(len(errors)), errors, alpha=0.7)
        ax4.set_xlabel('Time Index')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Absolute Error Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Convenience functions
def plot_predictions(data: pd.DataFrame, lr_model: Any, lstm_model: Any, 
                    preprocessor: Any, symbol: str = "Stock", test_size: float = 0.2) -> None:
    """Quick function to plot model predictions comparison"""
    try:
        visualizer = StockVisualizer()
        
        # Split data for testing
        split_idx = int(len(data) * (1 - test_size))
        test_data = data[split_idx:]
        
        # Prepare data for Linear Regression
        if hasattr(lr_model, 'is_trained') and lr_model.is_trained:
            X_test_lr, y_test_lr, test_dates = preprocessor.prepare_ml_data(test_data)
            lr_predictions = lr_model.predict(X_test_lr)
        else:
            lr_predictions = np.array([])
            print("⚠️ Linear Regression model not trained")
        
        # Prepare data for LSTM
        if hasattr(lstm_model, 'is_trained') and lstm_model.is_trained:
            X_test_lstm, y_test_lstm = preprocessor.prepare_lstm_data(test_data, time_steps=60)
            if len(X_test_lstm) > 0:
                lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
                lstm_predictions = preprocessor.inverse_transform_predictions(lstm_predictions_scaled)
                
                # Align dates for LSTM (account for time_steps offset)
                lstm_dates = test_data.index[60:60+len(lstm_predictions)]
                y_test_lstm_actual = preprocessor.inverse_transform_predictions(y_test_lstm)
            else:
                lstm_predictions = np.array([])
                print("⚠️ Insufficient data for LSTM predictions")
        else:
            lstm_predictions = np.array([])
            print("⚠️ LSTM model not trained")
        
        # Create comparison plot if we have predictions from both models
        if len(lr_predictions) > 0 and len(lstm_predictions) > 0:
            # Align data lengths (use shorter length)
            min_len = min(len(lr_predictions), len(lstm_predictions))
            
            # For LR, use the last min_len predictions
            lr_pred_aligned = lr_predictions[-min_len:]
            lr_actual_aligned = y_test_lr[-min_len:]
            lr_dates_aligned = test_dates[-min_len:]
            
            # For LSTM, use first min_len predictions
            lstm_pred_aligned = lstm_predictions[:min_len]
            lstm_actual_aligned = y_test_lstm_actual[:min_len]
            lstm_dates_aligned = lstm_dates[:min_len]
            
            # Use LSTM dates as they're more recent
            visualizer.plot_predictions_comparison(
                lstm_dates_aligned, lstm_actual_aligned, 
                lr_pred_aligned, lstm_pred_aligned, symbol
            )
            
            # Also plot individual accuracy plots
            visualizer.plot_prediction_accuracy(lr_actual_aligned, lr_pred_aligned, "Linear Regression")
            visualizer.plot_prediction_accuracy(lstm_actual_aligned, lstm_pred_aligned, "LSTM")
            
        elif len(lr_predictions) > 0:
            # Only LR predictions available
            visualizer.plot_prediction_accuracy(y_test_lr, lr_predictions, "Linear Regression")
            
        elif len(lstm_predictions) > 0:
            # Only LSTM predictions available
            visualizer.plot_prediction_accuracy(y_test_lstm_actual, lstm_predictions, "LSTM")
            
        else:
            print("❌ No trained models available for prediction plotting")
            
    except Exception as e:
        print(f"Error in plot_predictions: {e}")
        print("Use individual StockVisualizer methods for detailed plotting")

def plot_stock_overview(data: pd.DataFrame, symbol: str = "Stock") -> None:
    """Complete stock overview with multiple visualizations"""
    visualizer = StockVisualizer()
    
    print(f"📊 Generating comprehensive overview for {symbol}")
    
    # 1. Price history with moving averages
    columns_to_plot = ['Close']
    if 'MA_20' in data.columns:
        columns_to_plot.append('MA_20')
    if 'MA_50' in data.columns:
        columns_to_plot.append('MA_50')
    
    visualizer.plot_price_history(data, symbol, columns_to_plot)
    
    # 2. Volume analysis (if volume data available)
    if 'Volume' in data.columns:
        visualizer.plot_volume_analysis(data, symbol)
    
    # 3. Technical indicators (if available)
    technical_cols = ['MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']
    if any(col in data.columns for col in technical_cols):
        visualizer.plot_technical_indicators(data, symbol)
    
    # 4. Price distribution analysis
    visualizer.plot_price_distribution(data, symbol)
    
    # 5. Correlation matrix for numeric features
    if len(data.select_dtypes(include=[np.number]).columns) > 1:
        visualizer.plot_correlation_matrix(data, f"{symbol} Features Correlation")
    
    print("✅ Stock overview complete!")

def plot_complete_model_analysis(data: pd.DataFrame, lr_model: Any, lstm_model: Any, 
                                preprocessor: Any, symbol: str = "Stock") -> None:
    """Complete model analysis with all visualizations"""
    visualizer = StockVisualizer()
    
    print(f"🔍 Running complete model analysis for {symbol}")
    
    try:
        # 1. Model predictions comparison
        plot_predictions(data, lr_model, lstm_model, preprocessor, symbol)
        
        # 2. Feature importance for Linear Regression
        if hasattr(lr_model, 'is_trained') and lr_model.is_trained:
            importance_df = lr_model.get_feature_importance()
            if importance_df is not None:
                visualizer.plot_feature_importance(importance_df)
        
        # 3. Training history for LSTM
        if hasattr(lstm_model, 'history') and lstm_model.history is not None:
            visualizer.plot_training_loss(lstm_model.history.history, "LSTM Training History")
        
        # 4. Model metrics comparison
        if (hasattr(lr_model, 'metrics') and hasattr(lstm_model, 'metrics') and 
            lr_model.metrics and lstm_model.metrics):
            visualizer.plot_model_metrics_comparison(lr_model.metrics, lstm_model.metrics)
        
        print("✅ Complete model analysis finished!")
        
    except Exception as e:
        print(f"Error in complete model analysis: {e}")

def plot_future_analysis(data: pd.DataFrame, lr_model: Any, lstm_model: Any, 
                        preprocessor: Any, symbol: str = "Stock", days: int = 7) -> None:
    """Plot future predictions from both models"""
    visualizer = StockVisualizer()
    
    print(f"🔮 Generating future predictions for {symbol} ({days} days)")
    
    try:
        from datetime import timedelta
        import pandas as pd
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Linear Regression future predictions
        if hasattr(lr_model, 'is_trained') and lr_model.is_trained:
            try:
                X, _, _ = preprocessor.prepare_ml_data(data)
                lr_future = lr_model.predict_next_days(X[-1], days)
                
                visualizer.plot_future_predictions(
                    data, future_dates, lr_future, "Linear Regression", symbol
                )
            except Exception as e:
                print(f"⚠️ Linear Regression future prediction error: {e}")
        
        # LSTM future predictions
        if hasattr(lstm_model, 'is_trained') and lstm_model.is_trained:
            try:
                X_lstm, _ = preprocessor.prepare_lstm_data(data, time_steps=60)
                lstm_future = lstm_model.predict_next_days(
                    X_lstm[-1], days, preprocessor.scaler
                )
                
                visualizer.plot_future_predictions(
                    data, future_dates, lstm_future, "LSTM", symbol
                )
            except Exception as e:
                print(f"⚠️ LSTM future prediction error: {e}")
        
        print("✅ Future analysis complete!")
        
    except Exception as e:
        print(f"Error in future analysis: {e}")