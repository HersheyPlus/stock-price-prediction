from typing import Optional, Dict, Any
import yfinance as yf
import pandas as pd

class StockDataLoader:
    def __init__(self) -> None:
        self.data: Optional[pd.DataFrame] = None
        self.symbol: Optional[str] = None
    
    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load stock data from Yahoo Finance"""
        try:
            self.symbol = symbol
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(start=start_date, end=end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            print(f"Loaded {len(self.data)} records for {symbol}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_basic_info(self) -> Optional[Dict[str, Any]]:
        """Get basic stock information"""
        if self.data is None:
            return None
        
        info: Dict[str, Any] = {
            'symbol': self.symbol,
            'start_date': self.data.index[0],
            'end_date': self.data.index[-1],
            'total_records': len(self.data),
            'avg_close': float(self.data['Close'].mean()),
            'min_close': float(self.data['Close'].min()),
            'max_close': float(self.data['Close'].max()),
            'volatility': float(self.data['Close'].std())
        }
        return info
    
    def add_technical_indicators(self) -> Optional[pd.DataFrame]:
        """Add basic technical indicators"""
        if self.data is None:
            return None
        
        # Moving averages
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = self.data['Close'].rolling(window=20).mean()
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = rolling_mean + (rolling_std * 2)
        self.data['BB_lower'] = rolling_mean - (rolling_std * 2)
        
        return self.data

def load_stock_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Convenience function"""
    loader = StockDataLoader()
    return loader.load_stock_data(symbol, start_date, end_date)