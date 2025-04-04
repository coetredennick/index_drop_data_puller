import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_sp500_data(start_date, end_date):
    """
    Fetch S&P 500 historical data from Yahoo Finance
    
    Parameters:
    -----------
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing S&P 500 historical data
    """
    try:
        # Fetch data
        sp500 = yf.download(
            "^GSPC",
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Basic data cleaning
        if sp500.empty:
            return None
        
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
        for days, label in [(5, '1W'), (21, '1M'), (63, '3M'), (126, '6M'), (252, '1Y'), (756, '3Y')]:
            sp500[f'Fwd_Ret_{label}'] = sp500['Close'].pct_change(periods=days).shift(-days) * 100
        
        return sp500
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
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
