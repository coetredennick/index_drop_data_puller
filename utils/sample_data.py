import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def generate_sample_sp500_data(start_date='2020-01-01', end_date='2023-01-01'):
    """
    Generate sample S&P 500 data for demonstration purposes when Yahoo Finance API is unavailable
    
    Parameters:
    -----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample S&P 500 historical data
    """
    # Convert dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate daily date range (business days only)
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    # Set initial price
    initial_price = 3200  # Approximate S&P 500 value in early 2020
    
    # Create base dataframe with dates
    df = pd.DataFrame(index=date_range)
    
    # Generate prices with realistic volatility and trend
    n_days = len(date_range)
    
    # Parameters for price simulation
    annual_return = 0.08  # 8% average annual return
    annual_volatility = 0.20  # 20% annual volatility
    
    # Daily parameters
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    # Generate log returns with a slight positive drift
    np.random.seed(42)  # For reproducibility
    log_returns = np.random.normal(daily_return, daily_volatility, n_days)
    
    # Add some autocorrelation to make returns more realistic
    for i in range(1, len(log_returns)):
        log_returns[i] = 0.1 * log_returns[i-1] + 0.9 * log_returns[i]
    
    # Convert log returns to price series
    price_series = initial_price * np.exp(np.cumsum(log_returns))
    
    # Create OHLC data
    df['Open'] = price_series * (1 + np.random.normal(0, 0.003, n_days))
    df['High'] = np.maximum(price_series * (1 + np.random.normal(0.005, 0.005, n_days)), df['Open'])
    df['Low'] = np.minimum(price_series * (1 - np.random.normal(0.005, 0.005, n_days)), df['Open'])
    df['Close'] = price_series
    
    # Ensure High is always the highest and Low is always the lowest
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    # Add some volume data (in millions)
    base_volume = 4000  # Approximate volume in millions
    df['Volume'] = base_volume * (1 + np.random.normal(0, 0.3, n_days))
    
    # Volume tends to be higher on down days
    df.loc[log_returns < 0, 'Volume'] *= (1 + abs(log_returns[log_returns < 0]) * 5)
    
    # Volume in units (not millions)
    df['Volume'] = (df['Volume'] * 1_000_000).astype(int)
    
    # Add some market crash events for testing drop detection
    # Add a severe crash (simulating March 2020 COVID crash)
    covid_crash_start = pd.to_datetime('2020-02-20')
    if covid_crash_start in df.index:
        crash_duration = 22  # ~1 month of trading days
        
        # Find where to start the crash
        try:
            start_idx = df.index.get_loc(covid_crash_start)
            
            # Create a pattern of severe drops
            for i in range(crash_duration):
                if start_idx + i < len(df):
                    # More severe in the beginning
                    severity = 0.12 * np.exp(-i/10)
                    
                    # Don't drop every day, some days have rebounds
                    if i % 3 != 0:  # 2 out of 3 days have drops
                        current_price = df.iloc[start_idx + i - 1]['Close'] if i > 0 else df.iloc[start_idx]['Open']
                        
                        # Adjust current day prices for the drop
                        drop_factor = 1 - severity * (1 + 0.5 * np.random.random())
                        
                        # Update OHLC
                        df.iloc[start_idx + i, df.columns.get_loc('Close')] = current_price * drop_factor
                        df.iloc[start_idx + i, df.columns.get_loc('Open')] = current_price * (drop_factor + severity * 0.2)
                        df.iloc[start_idx + i, df.columns.get_loc('Low')] = current_price * (drop_factor - severity * 0.3)
                        df.iloc[start_idx + i, df.columns.get_loc('High')] = current_price * (drop_factor + severity * 0.4)
                        
                        # Higher volume during crash
                        df.iloc[start_idx + i, df.columns.get_loc('Volume')] *= (1.5 + severity * 5)
                    else:
                        # Small rebound days
                        current_price = df.iloc[start_idx + i - 1]['Close']
                        rebound = 1 + 0.02 * np.random.random()
                        
                        df.iloc[start_idx + i, df.columns.get_loc('Close')] = current_price * rebound
                        df.iloc[start_idx + i, df.columns.get_loc('Open')] = current_price * 0.99
                        df.iloc[start_idx + i, df.columns.get_loc('High')] = current_price * (rebound + 0.01)
                        df.iloc[start_idx + i, df.columns.get_loc('Low')] = current_price * 0.985
        except:
            # If date not in index, skip this modification
            pass
    
    # Add a few more single-day significant drops
    drop_dates = ['2020-06-11', '2021-09-28', '2022-05-18']
    for drop_date in drop_dates:
        try:
            date = pd.to_datetime(drop_date)
            if date in df.index:
                idx = df.index.get_loc(date)
                
                # Create a significant one-day drop (4-7%)
                drop_pct = 0.04 + 0.03 * np.random.random()
                
                prev_close = df.iloc[idx - 1]['Close'] if idx > 0 else df.iloc[idx]['Open']
                
                # Update the day's values
                df.iloc[idx, df.columns.get_loc('Close')] = prev_close * (1 - drop_pct)
                df.iloc[idx, df.columns.get_loc('Open')] = prev_close * 0.99
                df.iloc[idx, df.columns.get_loc('Low')] = prev_close * (1 - drop_pct - 0.01)
                df.iloc[idx, df.columns.get_loc('High')] = prev_close
                
                # Higher volume
                df.iloc[idx, df.columns.get_loc('Volume')] *= 2.5
        except:
            # If date not in index, skip this drop
            pass
    
    # Add calculated fields similar to real data
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change() * 100
    
    # Calculate daily price change
    df['Change'] = df['Close'] - df['Open']
    df['Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Calculate high-low range
    df['HL_Range'] = (df['High'] - df['Low']) / df['Open'] * 100
    
    # Forward returns for different time periods
    for days, label in [(1, '1D'), (2, '2D'), (3, '3D'), (5, '1W'), (21, '1M'), 
                        (63, '3M'), (126, '6M'), (252, '1Y')]:
        df[f'Fwd_Ret_{label}'] = df['Close'].pct_change(periods=days).shift(-days) * 100
    
    # Add simulated VIX data
    # Start with base VIX level
    base_vix = 15
    
    # VIX tends to be inversely related to market returns
    vix_series = base_vix + (-5 * log_returns * 100)
    
    # Add some randomness
    vix_series += np.random.normal(0, 1, n_days)
    
    # Ensure VIX is always positive and has realistic bounds
    vix_series = np.maximum(vix_series, 9)  # VIX floor
    
    # Higher spike during crash periods
    df['VIX_Close'] = vix_series
    
    # Spikes in VIX during market crashes
    df.loc[df['Return'] < -2, 'VIX_Close'] *= (1.2 + abs(df.loc[df['Return'] < -2, 'Return']) * 0.05)
    
    # Add OHLC for VIX
    df['VIX_Open'] = df['VIX_Close'].shift(1).fillna(df['VIX_Close'])
    df['VIX_High'] = df['VIX_Close'] * (1 + np.random.normal(0.02, 0.01, n_days))
    df['VIX_Low'] = df['VIX_Close'] * (1 - np.random.normal(0.02, 0.01, n_days))
    
    # VIX return
    df['VIX_Return'] = df['VIX_Close'].pct_change() * 100
    
    # Add a 5-day rolling average of VIX
    df['VIX_5D_Avg'] = df['VIX_Close'].rolling(window=5).mean()
    
    # Add a 20-day rolling average of VIX
    df['VIX_20D_Avg'] = df['VIX_Close'].rolling(window=20).mean()
    
    # Calculate VIX relative to its moving averages
    df['VIX_Rel_5D'] = (df['VIX_Close'] / df['VIX_5D_Avg'] - 1) * 100
    df['VIX_Rel_20D'] = (df['VIX_Close'] / df['VIX_20D_Avg'] - 1) * 100
    
    # Calculate high-low volatility range
    df['VIX_HL_Range'] = (df['VIX_High'] - df['VIX_Low']) / df['VIX_Open'] * 100
    
    # Clean up any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Make sure there are no NaN values left
    df = df.fillna(0)
    
    return df

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sample_sp500_data(start_date=None, end_date=None):
    """
    Get sample S&P 500 data for demonstration purposes
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in the format 'YYYY-MM-DD'
    end_date : str, optional
        End date in the format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample S&P 500 historical data
    """
    # Default dates if not provided
    if start_date is None:
        start_date = '2020-01-01'
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Generate sample data for a fixed period (slightly larger than requested)
    full_df = generate_sample_sp500_data('2020-01-01', '2023-12-31')
    
    # Filter to requested date range
    df = full_df.loc[start_date:end_date]
    
    if df.empty:
        # If no data in range, return None so app knows to show an error
        return None
    
    return df