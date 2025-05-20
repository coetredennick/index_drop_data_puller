import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_market_data(ticker_symbol, start_date, end_date, include_vix=True, max_retries=3, retry_delay=5):
    """
    Fetch market data from Yahoo Finance with retry mechanism
    
    Parameters:
    -----------
    ticker_symbol : str
        The ticker symbol to fetch (e.g., "^GSPC" for S&P 500, "^IXIC" for NASDAQ, "^DJI" for Dow Jones)
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    include_vix : bool, optional
        Whether to include VIX data (default: True)
    max_retries : int, optional
        Maximum number of retries for fetching data
    retry_delay : int, optional
        Delay between retries in seconds
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing market data with VIX data if requested
    """
    # For user-friendly error messages
    ticker_names = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones"
    }
    
    ticker_name = ticker_names.get(ticker_symbol, ticker_symbol)
    
    for attempt in range(max_retries):
        try:
            # Fetch data
            market_data = yf.download(
                ticker_symbol,
                start=start_date,
                end=end_date,
                progress=False,
                ignore_tz=True  # Added to help with timezone issues
            )
            
            # Basic data cleaning
            if market_data.empty:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)  # Wait before retrying
                    continue
                else:
                    st.warning(f"Could not fetch {ticker_name} data. The data source might be unavailable. Please try again later.")
                    return None
            
            # Fix for MultiIndex columns - flatten the columns
            if isinstance(market_data.columns, pd.MultiIndex):
                market_data.columns = [col[0] for col in market_data.columns]
                
            # Calculate daily returns
            market_data['Return'] = market_data['Close'].pct_change() * 100
            
            # Calculate daily price change
            market_data['Change'] = market_data['Close'] - market_data['Open']
            market_data['Change_Pct'] = (market_data['Close'] - market_data['Open']) / market_data['Open'] * 100
            
            # Calculate high-low range
            market_data['HL_Range'] = (market_data['High'] - market_data['Low']) / market_data['Open'] * 100
            
            # Forward returns for different time periods
            for days, label in [(1, '1D'), (2, '2D'), (3, '3D'), (5, '1W'), (21, '1M'), (63, '3M'), (126, '6M'), (252, '1Y'), (756, '3Y')]:
                market_data[f'Fwd_Ret_{label}'] = market_data['Close'].pct_change(periods=days).shift(-days) * 100
            
            # Fetch VIX data if requested
            if include_vix:
                try:
                    # Fetch VIX data with retries
                    vix_data = None
                    for vix_attempt in range(max_retries):
                        try:
                            vix_data = yf.download(
                                "^VIX",
                                start=start_date,
                                end=end_date,
                                progress=False,
                                ignore_tz=True
                            )
                            if not vix_data.empty:
                                break
                        except Exception as vix_e:
                            if vix_attempt < max_retries - 1:
                                import time
                                time.sleep(retry_delay)
                            else:
                                print(f"Failed to fetch VIX data after {max_retries} attempts: {vix_e}")
                    
                    if vix_data is not None and not vix_data.empty:
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
                        
                        # Join VIX data with market data
                        market_data = market_data.join(vix_data[[
                            'VIX_Close', 'VIX_Return', 'VIX_5D_Avg', 'VIX_20D_Avg', 
                            'VIX_Rel_5D', 'VIX_Rel_20D', 'VIX_HL_Range'
                        ]])
                        
                        # Fill any NaN values in VIX columns with forward filling, then backward filling
                        vix_cols = [col for col in market_data.columns if col.startswith('VIX_')]
                        if vix_cols:
                            # Use ffill() and bfill() methods instead of fillna(method=) to avoid deprecation warning
                            market_data[vix_cols] = market_data[vix_cols].ffill().bfill()
                    
                except Exception as vix_e:
                    # Log the error but continue without VIX data
                    print(f"Error fetching VIX data: {vix_e}")
                    # Continue without VIX data
            
            return market_data
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} failed for {ticker_name}: {e}. Retrying...")
                import time
                time.sleep(retry_delay)  # Wait before retrying
            else:
                st.warning(f"Failed to fetch {ticker_name} data after {max_retries} attempts. The data source might be rate-limited. Please try again later.")
                print(f"Error fetching {ticker_name} data: {e}")
                return None

# Maintain backwards compatibility with the old function name
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_sp500_data(start_date, end_date, include_vix=True):
    """Fetch S&P 500 data (wrapper for backwards compatibility)"""
    return fetch_market_data("^GSPC", start_date, end_date, include_vix)

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
