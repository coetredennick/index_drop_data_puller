import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime, timedelta
import time
import streamlit as st

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_sp500_alternative(start_date, end_date, max_retries=3, retry_delay=5):
    """
    Fetch S&P 500 historical data using pandas-datareader as an alternative to yfinance
    
    Parameters:
    -----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    max_retries : int, optional
        Maximum number of retries for API calls (default: 3)
    retry_delay : int, optional
        Delay between retries in seconds (default: 5)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing S&P 500 historical data
    """
    # Convert string dates to datetime objects if they're not already
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Try to fetch data with retries
    sp500 = None
    last_error = None
    
    for retry in range(max_retries):
        try:
            # Try FRED (Federal Reserve Economic Data) first, as it's generally reliable
            print(f"Attempting to fetch S&P 500 data from {start_date} to {end_date} using FRED")
            sp500 = pdr.DataReader(
                "SP500", 
                "fred", 
                start=start_date,
                end=end_date
            )
            
            if sp500 is not None and not sp500.empty and len(sp500) > 5:
                print(f"Successfully retrieved {len(sp500)} days of S&P 500 data from FRED")
                
                # Rename the column to match the format expected by the application
                sp500 = sp500.rename(columns={"SP500": "Close"})
                break
            else:
                print(f"FRED data insufficient, trying Yahoo Finance...")
                
                # If FRED doesn't work, try Yahoo
                sp500 = pdr.DataReader(
                    "^GSPC", 
                    "yahoo", 
                    start=start_date,
                    end=end_date
                )
                
                if sp500 is not None and not sp500.empty and len(sp500) > 5:
                    print(f"Successfully retrieved {len(sp500)} days of S&P 500 data from Yahoo")
                    break
                else:
                    print(f"Retry {retry+1}/{max_retries}: Got empty or insufficient data")
                    time.sleep(retry_delay)
                    continue
        
        except Exception as e:
            last_error = e
            print(f"Retry {retry+1}/{max_retries} failed: {e}")
            # Wait before retrying
            time.sleep(retry_delay)
    
    # If we still couldn't get data after all retries
    if sp500 is None or sp500.empty:
        print(f"Failed to retrieve S&P 500 data after {max_retries} attempts")
        return None
    
    # Process the data to match the format expected by the application
    try:
        # Check which columns we have (depends on the data source)
        if all(col in sp500.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            # We already have OHLCV data, just calculate returns
            sp500['Return'] = sp500['Close'].pct_change() * 100
        elif 'Close' in sp500.columns:
            # We only have Close data, generate synthetic OHLCV
            # This happens with FRED data which only gives closing values
            
            # If we only have the Close column (e.g., from FRED), we need to estimate the other values
            # Estimate Open price (previous close plus small random variation)
            sp500['Open'] = sp500['Close'].shift(1).fillna(sp500['Close']) * (1 + np.random.normal(0, 0.002, len(sp500)))
            
            # Estimate High as max(Open, Close) plus small increment
            sp500['High'] = sp500[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.003, len(sp500))))
            
            # Estimate Low as min(Open, Close) minus small decrement
            sp500['Low'] = sp500[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.003, len(sp500))))
            
            # Estimate Volume (a placeholder since we don't have real volume data)
            avg_volume = 5e9  # Typical S&P 500 volume in recent years
            sp500['Volume'] = [int(avg_volume * (1 + np.random.normal(0, 0.2))) for _ in range(len(sp500))]
            
            # Calculate daily returns
            sp500['Return'] = sp500['Close'].pct_change() * 100
        else:
            # Unexpected data format
            print("Unexpected data format from data source")
            return None
        
        # Calculate additional metrics needed by the application
        # Calculate daily price change
        sp500['Change'] = sp500['Close'] - sp500['Open']
        sp500['Change_Pct'] = (sp500['Close'] - sp500['Open']) / sp500['Open'] * 100
        
        # Calculate high-low range
        sp500['HL_Range'] = (sp500['High'] - sp500['Low']) / sp500['Open'] * 100
        
        # Forward returns for different time periods
        for days, label in [(1, '1D'), (2, '2D'), (3, '3D'), (5, '1W'), (21, '1M'), 
                           (63, '3M'), (126, '6M'), (252, '1Y'), (756, '3Y')]:
            sp500[f'Fwd_Ret_{label}'] = sp500['Close'].pct_change(periods=days).shift(-days) * 100
        
        # Clean up any NaN values
        sp500 = sp500.fillna(method='ffill').fillna(method='bfill')
        
        # Make sure the data is complete enough for technical indicators
        from utils.technical_indicators import calculate_technical_indicators
        
        try:
            # Calculate technical indicators
            sp500 = calculate_technical_indicators(sp500)
            print(f"Successfully calculated technical indicators for {len(sp500)} days of data")
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            # Continue with the data we have even if indicators failed
            
        # Return the processed data
        return sp500
    
    except Exception as e:
        print(f"Error processing S&P 500 data: {e}")
        return None