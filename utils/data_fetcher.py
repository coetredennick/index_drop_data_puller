import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os
import time
import random

# Enhanced version of the data fetcher with better error handling and retry logic
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_sp500_data(start_date, end_date, include_vix=True, max_retries=3, backoff_factor=2):
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
        Maximum number of retry attempts (default: 3)
    backoff_factor : int, optional
        Backoff factor for retry delay calculation (default: 2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing S&P 500 historical data with VIX data if requested
    """
    # Try to fetch S&P 500 data with retries
    sp500 = None
    error_msg = None
    
    for attempt in range(max_retries):
        try:
            st.info(f"Fetching S&P 500 data (attempt {attempt+1}/{max_retries})...")
            
            # Add jitter to avoid synchronized requests
            jitter = random.uniform(0.1, 1.0)
            
            # Fetch data with exponential backoff
            sp500 = yf.download(
                "^GSPC",
                start=start_date,
                end=end_date,
                progress=False,
                threads=False  # Disable threading to reduce rate limit issues
            )
            
            # If successful, break the retry loop
            if not sp500.empty:
                break
                
        except Exception as e:
            error_msg = str(e)
            # If rate limited, wait with exponential backoff before retrying
            if "Rate limit" in error_msg or "Too Many Requests" in error_msg:
                wait_time = (backoff_factor ** attempt) + jitter
                st.warning(f"Yahoo Finance rate limit reached. Waiting {wait_time:.1f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                # For other errors, also wait but with less backoff
                wait_time = attempt + jitter
                st.warning(f"Error fetching data: {error_msg}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    # If all retries failed, return None with error message
    if sp500 is None or sp500.empty:
        st.error(f"Failed to fetch S&P 500 data after {max_retries} attempts. Last error: {error_msg}")
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
        
        # Fetch VIX data if requested, with retries
        if include_vix:
            vix_data = None
            
            for attempt in range(max_retries):
                try:
                    st.info(f"Fetching VIX data (attempt {attempt+1}/{max_retries})...")
                    
                    # Add jitter to avoid synchronized requests
                    jitter = random.uniform(0.1, 1.0)
                    
                    # Fetch VIX data with exponential backoff
                    vix_data = yf.download(
                        "^VIX",
                        start=start_date,
                        end=end_date,
                        progress=False,
                        threads=False  # Disable threading to reduce rate limit issues
                    )
                    
                    # If successful, break the retry loop
                    if not vix_data.empty:
                        break
                        
                except Exception as e:
                    error_msg = str(e)
                    # If rate limited, wait with exponential backoff before retrying
                    if "Rate limit" in error_msg or "Too Many Requests" in error_msg:
                        wait_time = (backoff_factor ** attempt) + jitter
                        st.warning(f"Yahoo Finance rate limit reached for VIX data. Waiting {wait_time:.1f} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        # For other errors, also wait but with less backoff
                        wait_time = attempt + jitter
                        st.warning(f"Error fetching VIX data: {error_msg}. Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
            
            # If VIX data fetch was successful
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
                    st.warning(f"Error processing VIX data: {e}. Continuing without VIX data.")
            else:
                st.warning("Failed to fetch VIX data. Continuing without VIX data.")
        
        st.success("Data fetched successfully!")
        return sp500
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
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
    
    # Use the enhanced fetch function with retry mechanism
    return fetch_sp500_data(start_date, end_date, max_retries=3, backoff_factor=2)
