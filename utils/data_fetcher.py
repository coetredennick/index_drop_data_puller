import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_sp500_data(start_date, end_date, include_vix=True, max_retries=5, retry_delay=2):
    """
    Fetch S&P 500 historical data from Yahoo Finance with retry mechanism
    
    Parameters:
    -----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    include_vix : bool, optional
        Whether to include VIX data (default: True)
    max_retries : int, optional
        Maximum number of retries for API calls (default: 5)
    retry_delay : int, optional
        Delay between retries in seconds (default: 2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing S&P 500 historical data with VIX data if requested
    """
    import time
    
    # If we have cached data in session state, use it when API calls fail
    if 'fallback_sp500_data' in st.session_state and st.session_state.fallback_sp500_data is not None:
        fallback_data_available = True
    else:
        fallback_data_available = False
    
    # Try to fetch data with retries
    sp500 = None
    last_error = None
    
    for retry in range(max_retries):
        try:
            # Fetch S&P 500 data
            sp500 = yf.download(
                "^GSPC",
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if not sp500.empty:
                # We got data, break the retry loop
                break
            else:
                # Empty DataFrame, wait and retry
                time.sleep(retry_delay)
                continue
                
        except Exception as e:
            last_error = e
            # Wait before retrying (increasing delay with each retry)
            time.sleep(retry_delay * (retry + 1))
    
    # If we still couldn't get data after all retries
    if sp500 is None or sp500.empty:
        if fallback_data_available:
            # Use fallback data from session state
            st.warning("Using cached data due to API limitations. Results may not be up to date.")
            return st.session_state.fallback_sp500_data
        else:
            # No fallback data available
            error_msg = f"Failed to fetch S&P 500 data after {max_retries} attempts."
            if last_error:
                error_msg += f" Error: {last_error}"
            st.error(error_msg)
            return None
    
    try:
        # Basic data cleaning
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
            vix_data = None
            vix_last_error = None
            
            # Try to fetch VIX data with retries
            for retry in range(max_retries):
                try:
                    # Fetch VIX data
                    vix_data = yf.download(
                        "^VIX",
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not vix_data.empty:
                        # We got data, break the retry loop
                        break
                    else:
                        # Empty DataFrame, wait and retry
                        time.sleep(retry_delay)
                        continue
                        
                except Exception as e:
                    vix_last_error = e
                    # Wait before retrying (increasing delay with each retry)
                    time.sleep(retry_delay * (retry + 1))
            
            # Process VIX data if available
            if vix_data is not None and not vix_data.empty:
                try:
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
                        # Use ffill() and bfill() methods instead of fillna(method=) to avoid deprecation warning
                        sp500[vix_cols] = sp500[vix_cols].ffill().bfill()
                        
                except Exception as e:
                    # Log the error but continue without VIX data
                    print(f"Error processing VIX data: {e}")
            else:
                # Could not fetch VIX data after retries
                if vix_last_error:
                    print(f"Failed to fetch VIX data: {vix_last_error}")
                else:
                    print("Failed to fetch VIX data: Empty response")
        
        # Save successful data as fallback for future API failures
        st.session_state.fallback_sp500_data = sp500
        
        return sp500
        
    except Exception as e:
        st.error(f"Error processing S&P 500 data: {e}")
        
        if fallback_data_available:
            st.warning("Using cached data due to processing error. Results may not reflect the latest information.")
            return st.session_state.fallback_sp500_data
        
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
    
    return fetch_sp500_data(start_date, end_date)
