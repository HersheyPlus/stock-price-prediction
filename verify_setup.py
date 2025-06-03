#!/usr/bin/env python3
"""
Verify Complete Setup
Test all imports and basic functionality
"""

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        # Core data science
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        print("✅ Core data science packages")
        
        # TensorFlow and Keras
        import tensorflow as tf
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        print("✅ TensorFlow and Keras")
        
        # Optional packages
        try:
            import streamlit as st
            print("✅ Streamlit")
        except ImportError:
            print("⚠️  Streamlit (optional)")
        
        try:
            import prophet
            print("✅ Prophet")
        except ImportError:
            print("⚠️  Prophet (optional)")
        
        # Typing
        from typing import Optional, Dict, List, Tuple
        print("✅ Typing support")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test data loading
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        if not data.empty:
            print("✅ Yahoo Finance data loading")
        else:
            print("⚠️  Yahoo Finance returned empty data")
        
        # Test model creation
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
        
        model = Sequential()
        model.add(LSTM(50, input_shape=(60, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        print("✅ LSTM model creation")
        
        # Test data processing
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        test_data = np.random.random((100, 1))
        scaled = scaler.fit_transform(test_data)
        print("✅ Data preprocessing")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Stock Prediction Setup Verification")
    print("=" * 40)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 40)
    if imports_ok and functionality_ok:
        print("🎉 Setup verification successful!")
        print("Now run: python main.py")
    else:
        print("❌ Setup verification failed!")
        print("Please check the error messages above and install missing packages.")

if __name__ == "__main__":
    main()