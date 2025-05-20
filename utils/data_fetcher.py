import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os

# Dictionary mapping index names to their Yahoo Finance symbols and volatility indexes
INDEX_MAPPING = {
    "S&P 500": {
        "symbol": "^GSPC",
        "volatility_index": "^VIX",
        "description": "Standard & Poor's 500 - Large-cap U.S. stocks"
    },
    "Nasdaq": {
        "symbol": "^IXIC",
        "volatility_index": "^VXN",
        "description": "Nasdaq Composite - Tech-heavy U.S. stocks"
    },
    "Dow Jones": {
        "symbol": "^DJI",
        "volatility_index": "^VXD",
        "description": "Dow Jones Industrial Average - 30 large U.S. companies"
    },
    "Russell 2000": {
        "symbol": "^RUT",
        "volatility_index": "^RVX",
        "description": "Russell 2000 - Small-cap U.S. stocks"
    }
}

@st.cache_data(ttl=1800)  # Cache data for 30 minutes
def fetch_market_data(index_name="S&P 500", start_date=None, end_date=None, include_volatility=True, max_retries=3, retry_delay=5):
    """
    Fetch market index data from Yahoo Finance with improved robustness
    
    Parameters:
    -----------
    index_name : str
        Name of the index to fetch ("S&P 500", "Nasdaq", "Dow Jones", or "Russell 2000")
    start_date : str
        Start date in the format 'YYYY-MM-DD'
    end_date : str
        End date in the format 'YYYY-MM-DD'
    include_volatility : bool, optional
        Whether to include volatility index data (default: True)
    max_retries : int, optional
        Maximum number of download retries (default: 3)
    retry_delay : int, optional
        Delay between retries in seconds (default: 5)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing market index data with volatility data if requested
    """
    
    # For backward compatibility with code that calls fetch_sp500_data
    if index_name not in INDEX_MAPPING:
        st.warning(f"Unknown index: {index_name}. Defaulting to S&P 500.")
        index_name = "S&P 500"
    
    # Get the Yahoo Finance symbol for the selected index
    index_symbol = INDEX_MAPPING[index_name]["symbol"]
    volatility_symbol = INDEX_MAPPING[index_name]["volatility_index"]
    import time
    
    # Record the attempt count
    attempts = 0
    
    # If the date range is too large, split it into smaller chunks
    from datetime import datetime, timedelta
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Initialize market_data as None to handle cases where we don't execute the if or else blocks
    market_data = None
    
    # For all date ranges, use daily data but handle long periods differently
    if (end_dt - start_dt) > timedelta(days=365*5):  # Use chunking for periods over 5 years
        try:
            st.info(f"Fetching long-term {index_name} data in chunks to avoid rate limiting...")
            
            # Split into chunks of 1 year each
            chunks = []
            current_start = start_dt
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=365), end_dt)
                chunks.append((current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')))
                current_start = current_end + timedelta(days=1)
            
            # Fetch data for each chunk
            all_data = []
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                st.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
                
                # Add a delay between chunks to avoid rate limiting
                if i > 0:
                    time.sleep(retry_delay)
                
                chunk_attempts = 0
                while chunk_attempts < max_retries:
                    try:
                        chunk_data = yf.download(
                            index_symbol,
                            start=chunk_start,
                            end=chunk_end,
                            progress=False
                        )
                        
                        if not chunk_data.empty:
                            all_data.append(chunk_data)
                            break
                        else:
                            chunk_attempts += 1
                            st.warning(f"Received empty data for chunk. Retry {chunk_attempts}/{max_retries}...")
                            time.sleep(retry_delay)
                    
                    except Exception as e:
                        chunk_attempts += 1
                        error_msg = str(e)
                        
                        # Special handling for rate limit errors
                        if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                            st.warning(f"Yahoo Finance rate limit reached. Retry {chunk_attempts}/{max_retries} after delay...")
                            time.sleep(retry_delay * 3)  # Longer delay for rate limits
                        else:
                            st.error(f"Error fetching chunk: {e}. Retry {chunk_attempts}/{max_retries}...")
                            time.sleep(retry_delay)
                        
                        # If we've reached max retries for this chunk, continue to next chunk
                        if chunk_attempts >= max_retries:
                            st.error(f"Failed to fetch data for chunk {i+1} after {max_retries} attempts.")
                            break
            
            # Combine all chunks
            if all_data:
                market_data = pd.concat(all_data, axis=0)
                if not market_data.empty:
                    market_data = market_data[~market_data.index.duplicated(keep='first')]  # Remove duplicate indices
                    
                    # Fix for MultiIndex columns
                    if isinstance(market_data.columns, pd.MultiIndex):
                        market_data.columns = [col[0] for col in market_data.columns]
                    
                    st.success(f"Successfully fetched {index_name} data for {len(market_data)} trading days!")
                else:
                    st.warning(f"After combining chunks, no {index_name} data was available.")
                    return None
            else:
                st.warning(f"No {index_name} data available for the given date range.")
                return None
                
        except Exception as e:
            st.error(f"Error in chunked data fetch approach: {e}")
            # Continue with normal approach as fallback
            pass
    
    # If market_data is still None (chunking failed or wasn't used), try the normal approach
    if market_data is None or market_data.empty:
        # Normal approach with retries for rate limiting
        while attempts < max_retries:
            try:
                # Attempt to fetch data
                market_data = yf.download(
                    index_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # Break the loop if successful
                if not market_data.empty:
                    break
                
                # If we got empty data, try again
                attempts += 1
                st.warning(f"Received empty data. Retry {attempts}/{max_retries}...")
                time.sleep(retry_delay)
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                
                # Special handling for rate limit errors
                if "Rate limited" in error_msg or "Too Many Requests" in error_msg:
                    st.warning(f"Yahoo Finance rate limit reached. Retry {attempts}/{max_retries} after delay...")
                    time.sleep(retry_delay * 2)  # Longer delay for rate limits
                else:
                    st.error(f"Error fetching {index_name} data: {e}. Retry {attempts}/{max_retries}...")
                    time.sleep(retry_delay)
                
                # If we've reached max retries, return None
                if attempts >= max_retries:
                    st.error(f"Failed to fetch {index_name} data after {max_retries} attempts. Please try again later.")
                    return None
    
    # Basic data cleaning
    if market_data is None or market_data.empty:
        st.warning(f"No {index_name} data available for the given date range.")
        return None
    
    try:
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
        
        # Add index identifier field
        market_data['Index'] = index_name
        
        # Fetch volatility data if requested
        if include_volatility:
            vol_attempts = 0
            while vol_attempts < max_retries:
                try:
                    # Fetch volatility index data
                    vol_data = yf.download(
                        volatility_symbol,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not vol_data.empty:
                        # Fix for MultiIndex columns - flatten the columns
                        if isinstance(vol_data.columns, pd.MultiIndex):
                            vol_data.columns = [f'VOL_{col[0]}' for col in vol_data.columns]
                        else:
                            vol_data.columns = [f'VOL_{col}' for col in vol_data.columns]
                        
                        # Calculate volatility index daily changes and returns
                        vol_data['VOL_Return'] = vol_data['VOL_Close'].pct_change() * 100
                        
                        # Add rolling averages of volatility (used as features in ML models)
                        vol_data['VOL_5D_Avg'] = vol_data['VOL_Close'].rolling(window=5).mean()
                        vol_data['VOL_20D_Avg'] = vol_data['VOL_Close'].rolling(window=20).mean()
                        
                        # Calculate volatility relative to its moving averages
                        vol_data['VOL_Rel_5D'] = (vol_data['VOL_Close'] / vol_data['VOL_5D_Avg'] - 1) * 100
                        vol_data['VOL_Rel_20D'] = (vol_data['VOL_Close'] / vol_data['VOL_20D_Avg'] - 1) * 100
                        
                        # Calculate high-low volatility range
                        vol_data['VOL_HL_Range'] = (vol_data['VOL_High'] - vol_data['VOL_Low']) / vol_data['VOL_Open'] * 100
                        
                        # Join volatility data with market index data
                        market_data = market_data.join(vol_data[[
                            'VOL_Close', 'VOL_Return', 'VOL_5D_Avg', 'VOL_20D_Avg', 
                            'VOL_Rel_5D', 'VOL_Rel_20D', 'VOL_HL_Range'
                        ]])
                        
                        # Fill any NaN values in volatility columns with forward filling, then backward filling
                        vol_cols = [col for col in market_data.columns if col.startswith('VOL_')]
                        if vol_cols:
                            # Use ffill() and bfill() methods to avoid deprecation warning
                            market_data[vol_cols] = market_data[vol_cols].ffill().bfill()
                        
                        # Break the loop if successful
                        break
                    else:
                        # If we got empty data, try again
                        vol_attempts += 1
                        if vol_attempts < max_retries:
                            time.sleep(retry_delay)
                        else:
                            # After max retries, just continue without volatility data
                            st.info(f"Volatility data for {index_name} not available, continuing without volatility metrics.")
                    
                except Exception as e:
                    vol_attempts += 1
                    error_msg = str(e)
                    
                    # Special handling for rate limit errors
                    if vol_attempts < max_retries and ("Rate limited" in error_msg or "Too Many Requests" in error_msg):
                        time.sleep(retry_delay * 2)  # Longer delay for rate limits
                    else:
                        # After max retries or for other errors, just continue without volatility data
                        st.info(f"Volatility data for {index_name} not available (Error: {e}), continuing without volatility metrics.")
                        break
        
        return market_data
        
    except Exception as e:
        st.error(f"Error processing {index_name} data: {e}")
        return None

# Backward compatibility function for existing code
def fetch_sp500_data(start_date, end_date, include_vix=True, max_retries=3, retry_delay=5):
    """
    Backward compatibility function for existing code that uses fetch_sp500_data
    """
    return fetch_market_data(
        index_name="S&P 500",
        start_date=start_date,
        end_date=end_date,
        include_volatility=include_vix,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_market_components(index_name="S&P 500"):
    """
    Fetch the components of the selected market index
    
    Parameters:
    -----------
    index_name : str
        Name of the index to fetch components for
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing index components, or None if not available
    """
    try:
        if index_name == "S&P 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df
        elif index_name == "Dow Jones":
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            tables = pd.read_html(url)
            # The DJIA components are typically in the first table
            df = tables[1]  # May need to adjust based on Wikipedia page structure
            return df
        elif index_name == "Nasdaq":
            # For Nasdaq-100, not the full composite
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)
            df = tables[4]  # May need to adjust based on Wikipedia page structure
            return df
        elif index_name == "Russell 2000":
            # Russell 2000 has too many components to fetch easily
            # Return a message instead
            st.info("Russell 2000 has 2000 components, too many to display in the application.")
            return None
        else:
            st.warning(f"Components for {index_name} are not available.")
            return None
    except Exception as e:
        st.error(f"Error fetching {index_name} components: {e}")
        return None

# For backward compatibility
def get_sp500_components():
    """Fetch the current components of the S&P 500 index"""
    return get_market_components("S&P 500")

def cache_data(data, drop_events, consecutive_drop_events):
    """
    Cache calculated data to avoid recalculation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Market data with technical indicators
    drop_events : list
        List of single-day drop events
    consecutive_drop_events : list
        List of consecutive-day drop events
    """
    # This function doesn't need to do anything as we're using st.cache_data
    # But we'll keep it for future enhancements
    pass

def get_latest_market_data(index_name="S&P 500"):
    """
    Get the latest market data for the current market conditions tab
    
    Parameters:
    -----------
    index_name : str
        Name of the index to fetch data for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the latest market data (60 days)
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    return fetch_market_data(index_name, start_date, end_date)

# For backward compatibility
def get_latest_sp500_data():
    """
    Get the latest S&P 500 data for the current market conditions tab
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the latest S&P 500 data (60 days)
    """
    return get_latest_market_data("S&P 500")
