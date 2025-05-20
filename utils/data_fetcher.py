import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os

@st.cache_data(ttl=1800)  # Cache data for 30 minutes
def fetch_sp500_data(start_date, end_date, include_vix=True, max_retries=3, retry_delay=5):
    """
    Fetch S&P 500 historical data from Yahoo Finance with improved robustness
    
    Parameters:
    -----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    include_vix : bool, optional
        Whether to include VIX data (default: True)
    max_retries : int, optional
        Maximum number of download retries (default: 3)
    retry_delay : int, optional
        Delay between retries in seconds (default: 5)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing S&P 500 historical data with VIX data if requested
    """
    # Record the attempt count
    attempts = 0
    
    # If the date range is too large, split it into smaller chunks
    from datetime import datetime, timedelta
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # If the date range is more than 10 years, let's use a different strategy
    if (end_dt - start_dt) > timedelta(days=365*10):
        # Use a different interval for very long periods
        try:
            st.info("Fetching long-term data, this may take a moment...")
            sp500 = yf.download(
                "^GSPC",
                start=start_date,
                end=end_date,
                interval="1wk",  # Use weekly data for longer periods
                progress=False
            )
            
            if not sp500.empty:
                # Fix for MultiIndex columns
                if isinstance(sp500.columns, pd.MultiIndex):
                    sp500.columns = [col[0] for col in sp500.columns]
                
                # Resample to business days if needed
                # sp500 = sp500.asfreq('B', method='ffill')
                
                st.success("Successfully fetched long-term weekly data!")
            else:
                st.warning("No data available for the given date range.")
                return None
                
        except Exception as e:
            st.error(f"Error fetching long-term data: {e}")
            # Fallback to the normal approach
            pass
    else:
        # Normal approach with retries for rate limiting
        while attempts < max_retries:
            try:
                # Attempt to fetch data
                sp500 = yf.download(
                    "^GSPC",
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # Break the loop if successful
                if not sp500.empty:
                    break
                
                # If we got empty data, try again
                attempts += 1
                st.warning(f"Received empty data. Retry {attempts}/{max_retries}...")
                import time
                time.sleep(retry_delay)
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                
                # Special handling for rate limit errors
                if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                    st.warning(f"Yahoo Finance rate limit reached. Retry {attempts}/{max_retries} after delay...")
                    import time
                    time.sleep(retry_delay * 2)  # Longer delay for rate limits
                else:
                    st.error(f"Error fetching S&P 500 data: {e}. Retry {attempts}/{max_retries}...")
                    import time
                    time.sleep(retry_delay)
                
                # If we've reached max retries, return None
                if attempts >= max_retries:
                    st.error(f"Failed to fetch S&P 500 data after {max_retries} attempts. Please try again later.")
                    return None
    
    # Basic data cleaning
    if sp500 is None or sp500.empty:
        st.warning("No S&P 500 data available for the given date range.")
        return None
    
    try:
        # Fix for MultiIndex columns - flatten the columns
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = [col[0] for col in sp500.columns]
            
        # Calculate daily returns
        sp500['Return'] = sp500['Close'].pct_change() * 100
        
        # Calculate daily price change
        sp500['Change'] = sp500['Close'] - sp500['Open']
        sp500['Change_Pct'] = (sp500['Close'] - sp500['Open']) / sp500['Open'] * 100
        
        # Calculate high-low range
        sp500['HL_Range'] = (sp500['High'] - sp500['Low']) / sp500['Open'] * 100
        
        # Forward returns for different time periods
        for days, label in [(1, '1D'), (2, '2D'), (3, '3D'), (5, '1W'), (21, '1M'), (63, '3M'), (126, '6M'), (252, '1Y'), (756, '3Y')]:
            sp500[f'Fwd_Ret_{label}'] = sp500['Close'].pct_change(periods=days).shift(-days) * 100
        
        # Fetch VIX data if requested
        if include_vix:
            vix_attempts = 0
            while vix_attempts < max_retries:
                try:
                    # Fetch VIX data
                    vix_data = yf.download(
                        "^VIX",
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not vix_data.empty:
                        # Fix for MultiIndex columns - flatten the columns
                        if isinstance(vix_data.columns, pd.MultiIndex):
                            vix_data.columns = [f'VIX_{col[0]}' for col in vix_data.columns]
                        else:
                            vix_data.columns = [f'VIX_{col}' for col in vix_data.columns]
                        
                        # Calculate VIX daily changes and returns
                        vix_data['VIX_Return'] = vix_data['VIX_Close'].pct_change() * 100
                        
                        # Add a 5-day rolling average of VIX (used as a feature in ML models)
                        vix_data['VIX_5D_Avg'] = vix_data['VIX_Close'].rolling(window=5).mean()
                        
                        # Add a 20-day rolling average of VIX (used as a feature in ML models)
                        vix_data['VIX_20D_Avg'] = vix_data['VIX_Close'].rolling(window=20).mean()
                        
                        # Calculate VIX relative to its moving averages
                        vix_data['VIX_Rel_5D'] = (vix_data['VIX_Close'] / vix_data['VIX_5D_Avg'] - 1) * 100
                        vix_data['VIX_Rel_20D'] = (vix_data['VIX_Close'] / vix_data['VIX_20D_Avg'] - 1) * 100
                        
                        # Calculate high-low volatility range
                        vix_data['VIX_HL_Range'] = (vix_data['VIX_High'] - vix_data['VIX_Low']) / vix_data['VIX_Open'] * 100
                        
                        # Join VIX data with S&P 500 data
                        sp500 = sp500.join(vix_data[[
                            'VIX_Close', 'VIX_Return', 'VIX_5D_Avg', 'VIX_20D_Avg', 
                            'VIX_Rel_5D', 'VIX_Rel_20D', 'VIX_HL_Range'
                        ]])
                        
                        # Fill any NaN values in VIX columns with forward filling, then backward filling
                        vix_cols = [col for col in sp500.columns if col.startswith('VIX_')]
                        if vix_cols:
                            # Use ffill() and bfill() methods to avoid deprecation warning
                            sp500[vix_cols] = sp500[vix_cols].ffill().bfill()
                        
                        # Break the loop if successful
                        break
                    else:
                        # If we got empty data, try again
                        vix_attempts += 1
                        if vix_attempts < max_retries:
                            import time
                            time.sleep(retry_delay)
                        else:
                            # After max retries, just continue without VIX data
                            st.info("VIX data not available, continuing without volatility metrics.")
                    
                except Exception as e:
                    vix_attempts += 1
                    error_msg = str(e)
                    
                    # Special handling for rate limit errors
                    if vix_attempts < max_retries and ("Rate limited" in error_msg or "Too Many Requests" in error_msg):
                        import time
                        time.sleep(retry_delay * 2)  # Longer delay for rate limits
                    else:
                        # After max retries or for other errors, just continue without VIX data
                        st.info(f"VIX data not available (Error: {e}), continuing without volatility metrics.")
                        break
        
        return sp500
        
    except Exception as e:
        st.error(f"Error processing S&P 500 data: {e}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sp500_components():
    """Fetch the current components of the S&P 500 index"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        return df
    except:
        return None

def cache_data(data, drop_events, consecutive_drop_events):
    """
    Cache calculated data to avoid recalculation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        S&P 500 data with technical indicators
    drop_events : list
        List of single-day drop events
    consecutive_drop_events : list
        List of consecutive-day drop events
    """
    # This function doesn't need to do anything as we're using st.cache_data
    # But we'll keep it for future enhancements
    pass

def create_sample_data(start_date, end_date):
    """
    Create sample S&P 500 data when API fails
    
    Parameters:
    -----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample S&P 500 data
    """
    st.warning("Using sample data for demonstration purposes. Real data could not be fetched from Yahoo Finance.")
    
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Create a date range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='B')  # Business days
    
    # Create a sample dataset with realistic S&P 500 patterns
    np.random.seed(42)  # For reproducibility
    
    # Start with a base value around 4000 (typical S&P 500 value in recent years)
    base_price = 4000
    
    # Create prices with realistic volatility
    daily_returns = np.random.normal(0.0005, 0.015, len(date_range))  # Mean return and volatility
    prices = [base_price]
    
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]  # Remove the first base price
    
    # Create a sample DataFrame
    sample_data = pd.DataFrame({
        'Open': prices * np.random.uniform(0.995, 1.000, len(prices)),
        'High': prices * np.random.uniform(1.005, 1.015, len(prices)),
        'Low': prices * np.random.uniform(0.985, 0.995, len(prices)),
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(2000000000, 5000000000, len(prices))
    }, index=date_range)
    
    # Calculate returns and other metrics
    sample_data['Return'] = sample_data['Close'].pct_change() * 100
    sample_data['Change'] = sample_data['Close'] - sample_data['Open']
    sample_data['Change_Pct'] = (sample_data['Close'] - sample_data['Open']) / sample_data['Open'] * 100
    sample_data['HL_Range'] = (sample_data['High'] - sample_data['Low']) / sample_data['Open'] * 100
    
    # Forward returns for different time periods
    for days, label in [(1, '1D'), (2, '2D'), (3, '3D'), (5, '1W'), (21, '1M'), (63, '3M'), (126, '6M'), (252, '1Y')]:
        if days < len(sample_data):
            sample_data[f'Fwd_Ret_{label}'] = sample_data['Close'].pct_change(periods=days).shift(-days) * 100
    
    # Create synthetic VIX data (typically inversely correlated with S&P)
    vix_base = 20
    sample_data['VIX_Close'] = vix_base - sample_data['Return'] * 0.5 + np.random.normal(0, 2, len(sample_data))
    sample_data['VIX_Close'] = sample_data['VIX_Close'].clip(lower=10, upper=40)  # Realistic VIX range
    
    sample_data['VIX_Return'] = sample_data['VIX_Close'].pct_change() * 100
    sample_data['VIX_5D_Avg'] = sample_data['VIX_Close'].rolling(window=5).mean()
    sample_data['VIX_20D_Avg'] = sample_data['VIX_Close'].rolling(window=20).mean()
    
    # Add RSI and other indicators to make it more complete
    sample_data['RSI_14'] = 50 + (sample_data['Return'].rolling(window=14).mean() * 5)
    sample_data['RSI_14'] = sample_data['RSI_14'].clip(lower=30, upper=70)
    
    return sample_data

def get_latest_sp500_data():
    """
    Get the latest S&P 500 data for the current market conditions tab
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the latest S&P 500 data (30 days)
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    data = fetch_sp500_data(start_date, end_date)
    
    # If we couldn't fetch real data, use sample data for demonstration
    if data is None or data.empty:
        return create_sample_data(start_date, end_date)
    
    return data
