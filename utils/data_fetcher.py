import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os
import time
import random

# Mapping for main indices to their volatility counterparts
VOLATILITY_MAP = {
    "^GSPC": {"symbol": "^VIX", "name": "VIX", "prefix": "VIX"},         # S&P 500 uses VIX
    "^IXIC": {"symbol": "^VIX", "name": "VIX", "prefix": "VIX"},         # NASDAQ uses VIX
    "^DJI": {"symbol": "^VIX", "name": "VIX", "prefix": "VIX"},          # Dow Jones uses VIX
    "^RUT": {"symbol": "^VIX", "name": "VIX", "prefix": "VIX"}          # Russell 2000 uses VIX
}

# Enhanced version of the data fetcher with better error handling and retry logic
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_index_data(index_symbol, index_name, start_date, end_date, include_volatility_data=True, max_retries=3, backoff_factor=2):
    """
    Fetch historical data for a given market index from Yahoo Finance with retry mechanism.
    Optionally includes corresponding volatility index data, with columns named using their specific prefixes (e.g., 'VIX_Close').
    
    Parameters:
    -----------
    index_symbol : str
        Yahoo Finance ticker for the main index (e.g., "^GSPC", "^IXIC")
    index_name : str
        Display name for the index (e.g., "S&P 500", "Nasdaq Composite")
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    include_volatility_data : bool, optional
        Whether to include corresponding volatility index data (default: True)
    max_retries : int, optional
        Maximum number of retry attempts (default: 3)
    backoff_factor : int, optional
        Backoff factor for retry delay calculation (default: 2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing historical data for the specified index, potentially with volatility data.
    """
    # Try to fetch main index data with retries
    main_data = None
    error_msg = None
    
    for attempt in range(max_retries):
        try:
            st.info(f"Fetching {index_name} data (Symbol: {index_symbol}, Attempt {attempt+1}/{max_retries})...")
            
            jitter = random.uniform(0.1, 1.0)
            
            main_data = yf.download(
                index_symbol,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False
            )
            
            if not main_data.empty:
                break
                
        except Exception as e:
            error_msg = str(e)
            if "Rate limit" in error_msg or "Too Many Requests" in error_msg:
                wait_time = (backoff_factor ** attempt) + jitter
                st.warning(f"Yahoo Finance rate limit reached for {index_name}. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                wait_time = attempt + jitter
                st.warning(f"Error fetching {index_name} data: {error_msg}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
    
    if main_data is None or main_data.empty:
        st.error(f"Failed to fetch {index_name} data (Symbol: {index_symbol}) after {max_retries} attempts. Last error: {error_msg}")
        return None
    
    try:
        if isinstance(main_data.columns, pd.MultiIndex):
            main_data.columns = [col[0] for col in main_data.columns]
            
        main_data['Return'] = main_data['Close'].pct_change() * 100
        main_data['Change'] = main_data['Close'] - main_data['Open']
        main_data['Change_Pct'] = (main_data['Close'] - main_data['Open']) / main_data['Open'] * 100
        main_data['HL_Range'] = (main_data['High'] - main_data['Low']) / main_data['Open'] * 100
        
        for days, label in [(1, '1D'), (2, '2D'), (3, '3D'), (5, '1W'), (21, '1M'), (63, '3M'), (126, '6M'), (252, '1Y'), (756, '3Y')]:
            main_data[f'Fwd_Ret_{label}'] = main_data['Close'].pct_change(periods=days).shift(-days) * 100
        
        if include_volatility_data:
            vol_map_entry = VOLATILITY_MAP.get(index_symbol)
            if vol_map_entry:
                vol_symbol = vol_map_entry["symbol"]
                vol_name = vol_map_entry["name"]
                vol_prefix = vol_map_entry["prefix"] # Original prefix like VIX, VXN
                
                volatility_data_df = None
                for attempt in range(max_retries):
                    try:
                        st.info(f"Fetching {vol_name} data (Symbol: {vol_symbol}, Attempt {attempt+1}/{max_retries})...")
                        jitter = random.uniform(0.1, 1.0)
                        volatility_data_df = yf.download(
                            vol_symbol,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            threads=False
                        )
                        if not volatility_data_df.empty:
                            break
                    except Exception as e:
                        error_msg = str(e)
                        if "Rate limit" in error_msg or "Too Many Requests" in error_msg:
                            wait_time = (backoff_factor ** attempt) + jitter
                            st.warning(f"Yahoo Finance rate limit reached for {vol_name}. Waiting {wait_time:.1f}s...")
                            time.sleep(wait_time)
                        else:
                            wait_time = attempt + jitter
                            st.warning(f"Error fetching {vol_name} data: {error_msg}. Retrying in {wait_time:.1f}s...")
                            time.sleep(wait_time)
                
                if volatility_data_df is not None and not volatility_data_df.empty:
                    try:
                        if isinstance(volatility_data_df.columns, pd.MultiIndex):
                            volatility_data_df.columns = [f'{vol_prefix}_{col[0]}' for col in volatility_data_df.columns]
                        else:
                            volatility_data_df.columns = [f'{vol_prefix}_{col}' for col in volatility_data_df.columns]
                        
                        # Use original prefix for calculations specific to this vol index
                        vol_close_col = f'{vol_prefix}_Close'
                        vol_open_col = f'{vol_prefix}_Open'
                        vol_high_col = f'{vol_prefix}_High'
                        vol_low_col = f'{vol_prefix}_Low'

                        volatility_data_df[f'{vol_prefix}_Return'] = volatility_data_df[vol_close_col].pct_change() * 100
                        volatility_data_df[f'{vol_prefix}_5D_Avg'] = volatility_data_df[vol_close_col].rolling(window=5).mean()
                        volatility_data_df[f'{vol_prefix}_20D_Avg'] = volatility_data_df[vol_close_col].rolling(window=20).mean()
                        volatility_data_df[f'{vol_prefix}_Rel_5D'] = (volatility_data_df[vol_close_col] / volatility_data_df[f'{vol_prefix}_5D_Avg'] - 1) * 100
                        volatility_data_df[f'{vol_prefix}_Rel_20D'] = (volatility_data_df[vol_close_col] / volatility_data_df[f'{vol_prefix}_20D_Avg'] - 1) * 100
                        if vol_open_col in volatility_data_df and vol_high_col in volatility_data_df and vol_low_col in volatility_data_df:
                             volatility_data_df[f'{vol_prefix}_HL_Range'] = (volatility_data_df[vol_high_col] - volatility_data_df[vol_low_col]) / volatility_data_df[vol_open_col] * 100
                        else:
                            volatility_data_df[f'{vol_prefix}_HL_Range'] = np.nan

                        # Columns to select from volatility_data_df (already correctly prefixed)
                        cols_to_select = []
                        if f'{vol_prefix}_Close' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_Close')
                        if f'{vol_prefix}_Return' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_Return')
                        if f'{vol_prefix}_5D_Avg' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_5D_Avg')
                        if f'{vol_prefix}_20D_Avg' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_20D_Avg')
                        if f'{vol_prefix}_Rel_5D' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_Rel_5D')
                        if f'{vol_prefix}_Rel_20D' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_Rel_20D')
                        if f'{vol_prefix}_HL_Range' in volatility_data_df: cols_to_select.append(f'{vol_prefix}_HL_Range')
                        
                        if cols_to_select: # ensure there are columns to join
                            selected_vol_data = volatility_data_df[cols_to_select]
                            main_data = main_data.join(selected_vol_data)
                            
                            # Forward fill and backward fill for the joined volatility columns
                            main_data[cols_to_select] = main_data[cols_to_select].ffill().bfill()
                        
                    except Exception as e:
                        st.warning(f"Error processing {vol_name} data: {e}. Continuing without {vol_name} data.")
                else:
                    st.warning(f"Failed to fetch {vol_name} data. Continuing without its data.")
            elif index_symbol not in VOLATILITY_MAP:
                 st.info(f"No volatility index mapping configured for {index_name} (Symbol: {index_symbol}). Proceeding without volatility data.")
        
        st.success(f"{index_name} data fetched and processed successfully!")
        return main_data
        
    except Exception as e:
        st.error(f"Error processing {index_name} data: {e}")
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
    return fetch_index_data("^GSPC", "S&P 500", start_date, end_date, max_retries=3, backoff_factor=2)
