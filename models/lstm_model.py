from typing import Optional, Dict, List, Tuple
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMModel:
    def __init__(self, time_steps: int = 60, features: int = 1) -> None:
        self.time_steps = time_steps
        self.features = features
        self.model: Optional[Sequential] = None
        self.history: Optional[History] = None
        self.is_trained: bool = False
        self.metrics: Dict[str, float] = {}
    
    def build_model(self, lstm_units: List[int] = [50, 50], dropout: float = 0.2, learning_rate: float = 0.001) -> Sequential:
        """Build LSTM model architecture"""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=(self.time_steps, self.features)
        ))
        self.model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            return_sequences = i < len(lstm_units) - 2  # Only last layer doesn't return sequences
            self.model.add(LSTM(units=units, return_sequences=return_sequences))
            self.model.add(Dropout(dropout))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("LSTM Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              epochs: int = 100, batch_size: int = 32, verbose: int = 1) -> bool:
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()
        
        # Reshape input data
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.features)
        
        # Validation data
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if X_val is not None and y_val is not None:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], self.features)
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        try:
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=False  # Don't shuffle time series data
            )
            
            self.is_trained = True
            
            # Calculate training metrics
            train_pred = self.predict(X_train.reshape(X_train.shape[0], X_train.shape[1]))
            self.metrics['train_mse'] = mean_squared_error(y_train, train_pred)
            self.metrics['train_rmse'] = np.sqrt(self.metrics['train_mse'])
            self.metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
            
            print(f"Training completed!")
            print(f"Training RMSE: {self.metrics['train_rmse']:.4f}")
            print(f"Training MAE: {self.metrics['train_mae']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = X.reshape(X.shape[0], X.shape[1], self.features)
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        metrics: Dict[str, float] = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        self.metrics.update({f'test_{k}': v for k, v in metrics.items()})
        
        return metrics, predictions
    
    def predict_next_days(self, last_sequence: np.ndarray, days: int = 5, 
                         scaler: Optional[MinMaxScaler] = None) -> np.ndarray:
        """Predict next N days"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next value
            next_pred = self.predict(current_sequence.reshape(1, -1))[0]
            predictions.append(next_pred)
            
            # Update sequence (remove first element, add prediction)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        predictions = np.array(predictions)
        
        # Inverse transform if scaler provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def plot_training_history(self) -> None:
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        from keras.models import load_model
        self.model = load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        if not self.is_trained:
            return "Model not trained yet"
        
        summary = f"""
            LSTM Model Summary:
            ==================
            Time Steps: {self.time_steps}
            Features: {self.features}
            Training RMSE: {self.metrics.get('train_rmse', 'N/A'):.4f}
            Training MAE: {self.metrics.get('train_mae', 'N/A'):.4f}
            Test RMSE: {self.metrics.get('test_rmse', 'N/A'):.4f}
            Test MAE: {self.metrics.get('test_mae', 'N/A'):.4f}
            """
        return summary