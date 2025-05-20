import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.data_fetcher import fetch_sp500_data, fetch_market_data, cache_data
from utils.technical_indicators import calculate_technical_indicators
from utils.event_detection import detect_drop_events, detect_consecutive_drops
from pages.historical_performance import show_historical_performance
from pages.ml_predictions_new import show_ml_predictions

# Configure the page
st.set_page_config(
    page_title="S&P 500 Market Drop Analyzer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"  # Explicitly collapse the sidebar
)

# Add CSS for clean, modern financial dashboard design
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Overall font styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Hide the default Streamlit navigation sidebar and all related elements */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide the hamburger menu that shows the navigation sidebar - stronger selector */
    header[data-testid="stHeader"] > div:first-child > div:first-child {
        display: none !important;
    }
    
    /* Hide the hamburger menu completely */
    button[kind="header"], 
    *[data-testid="collapsedControl"],
    *[data-testid="expandedControl"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        pointer-events: none !important;
    }
    
    /* Clean up left padding now that the sidebar toggle is gone */
    header[data-testid="stHeader"] {
        padding-left: 1rem !important;
    }
    
    /* Main header styling */
    h1 {
        color: #0D2535;
        font-weight: 600;
        font-size: 28px;
        margin-bottom: 0.2em;
    }
    
    h2, h3, h4 {
        color: #0D2535;
        font-weight: 500;
    }
    
    h3 {
        font-size: 1.1em;
        margin-top: 1em;
        margin-bottom: 0.5em;
        padding-bottom: 0.2em;
        border-bottom: 1px solid #f0f2f6;
    }

    /* Tab styling for clean navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 5px 5px 0 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 15px;
        font-size: 14px;
        font-weight: 500;
        color: #0D2535;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #1E88E5;
        color: #1E88E5;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.3rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1976D2;
    }
    
    /* Tables & Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    
    /* Settings container styling */
    div[data-testid="stExpander"] {
        border: 1px solid #f0f2f6;
        border-radius: 5px;
        margin-bottom: 1rem;
    }

    /* Reduce whitespace */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) {
        margin-bottom: 0;
    }
    
    /* Slider refinements */
    div[data-testid="stSlider"] {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    
    /* Make plotly charts responsive */
    iframe {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Cleaner title and description layout
st.markdown("""
<div style="text-align: center; padding: 1rem 0; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 1rem;">
    <h1 style="margin: 0; padding: 0; color: #1E4A7B;">S&P 500 Market Drop Analyzer</h1>
    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #5A6570;">
        A data-driven tool for analyzing market corrections and forecasting recovery patterns
    </p>
</div>
"""
, unsafe_allow_html=True)

# Initialize session state for settings
if 'drop_threshold' not in st.session_state:
    st.session_state.drop_threshold = 0.1
if 'consecutive_days' not in st.session_state:
    st.session_state.consecutive_days = 1
if 'date_range' not in st.session_state:
    st.session_state.date_range = ('1990-01-01', datetime.today().strftime('%Y-%m-%d'))
if 'data' not in st.session_state:
    st.session_state.data = None
if 'drop_events' not in st.session_state:
    st.session_state.drop_events = None
if 'consecutive_drop_events' not in st.session_state:
    st.session_state.consecutive_drop_events = None
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None
if 'current_event_type_filter' not in st.session_state:
    st.session_state.current_event_type_filter = 'all'
    
# Initialize data for each market index
for index in ['sp500', 'nasdaq', 'dow']:
    if f'{index}_data' not in st.session_state:
        st.session_state[f'{index}_data'] = None
    if f'{index}_drop_events' not in st.session_state:
        st.session_state[f'{index}_drop_events'] = None
    if f'{index}_consecutive_drop_events' not in st.session_state:
        st.session_state[f'{index}_consecutive_drop_events'] = None
        
# Track which index tab is active and has been loaded
if 'active_index' not in st.session_state:
    st.session_state.active_index = 'sp500'  # Default to S&P 500
if 'loaded_indices' not in st.session_state:
    st.session_state.loaded_indices = set()  # Track which indices have been loaded

# Main page settings in a clean container
# Use a form to prevent reloads when adjusting sliders
with st.form(key="analysis_settings_form"):
    # Use two columns for a cleaner layout - left for dates, right for thresholds
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h4 style='font-size: 1rem; margin-bottom: 0.7rem;'>Date Range</h4>", unsafe_allow_html=True)
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(st.session_state.date_range[0]),
                min_value=pd.to_datetime('1950-01-01'),
                max_value=datetime.today() - timedelta(days=1),
                key="start_date"
            )
        
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(st.session_state.date_range[1]),
                min_value=pd.to_datetime('1950-01-01'),
                max_value=datetime.today(),
                key="end_date"
            )
    
    with col2:
        st.markdown("<h4 style='font-size: 1rem; margin-bottom: 0.7rem;'>Drop Event Detection</h4>", unsafe_allow_html=True)
        
        drop_threshold = st.slider(
            "Drop Threshold (%)",
            min_value=0.1,
            max_value=20.0,
            value=st.session_state.drop_threshold,
            step=0.1
        )
        
        detection_col1, detection_col2 = st.columns([3, 2])
        
        with detection_col1:
            use_consecutive = st.checkbox(
                "Detect Consecutive Drops",
                value=st.session_state.consecutive_days > 1
            )
        
        with detection_col2:
            consecutive_days = 1
            if use_consecutive:
                consecutive_days = st.number_input(
                    "Days",
                    min_value=2,
                    max_value=5,
                    value=max(2, st.session_state.consecutive_days)
                )
    
    # Form submit button
    submit_button = st.form_submit_button("Apply Settings", use_container_width=True)

# Process form submission outside the form block
if submit_button:
    # Update session state
    st.session_state.drop_threshold = drop_threshold
    st.session_state.consecutive_days = consecutive_days if use_consecutive else 1
    st.session_state.date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Reset data for all indices
    for index in ['sp500', 'nasdaq', 'dow']:
        st.session_state[f'{index}_data'] = None
        st.session_state[f'{index}_drop_events'] = None
        st.session_state[f'{index}_consecutive_drop_events'] = None
    
    # Also reset the legacy variables
    st.session_state.data = None
    st.session_state.drop_events = None
    st.session_state.consecutive_drop_events = None
    st.session_state.selected_event = None
    
    # Clear loaded indices
    st.session_state.loaded_indices = set()
    
    # Show info message
    st.success("✅ Settings applied! Data will be refreshed.")
    st.rerun()

# Data source info with improved styling
st.markdown("""
<div style="text-align: right; font-size: 0.8em; color: #5A6570; margin-top: -0.5rem; margin-bottom: 0.7rem;">
    <span style="background-color: #f0f2f6; padding: 0.2rem 0.5rem; border-radius: 3px;">
        <i>Data source: Yahoo Finance (^GSPC)</i>
    </span>
</div>
""", unsafe_allow_html=True)

# We now fetch data on-demand in each tab instead of pre-loading it here
# This allows users to only load data for markets they're interested in

# Create market index tabs with icons for better visual organization
market_tabs = st.tabs([
    "📈 S&P 500", 
    "📊 NASDAQ",
    "💼 Dow Jones"
])

# Define market ticker mapping
market_tickers = {
    0: {"name": "S&P 500", "ticker": "^GSPC", "key": "sp500"},
    1: {"name": "NASDAQ", "ticker": "^IXIC", "key": "nasdaq"},
    2: {"name": "Dow Jones", "ticker": "^DJI", "key": "dow"}
}

# Add a light separator before tabs content
st.markdown('<hr style="margin-top: 0; margin-bottom: 15px; border: none; height: 1px; background-color: #f0f2f6;">', unsafe_allow_html=True)

# Handle each market index tab
for i, tab in enumerate(market_tabs):
    with tab:
        market_info = market_tickers[i]
        market_name = market_info["name"]
        market_ticker = market_info["ticker"]
        market_key = market_info["key"]
        
        # Update active index
        st.session_state.active_index = market_key
        
        # Display market specific heading
        st.markdown(f"### {market_name} Market Drop Analysis")
        
        # Create sub-tabs for different analysis views
        sub_tabs = st.tabs([
            "📈 Historical Performance", 
            "🤖 ML Predictions"
        ])
        
        # Check if we need to fetch data for this index
        with st.spinner(f"Fetching {market_name} data..."):
            # Only load data if this tab is active and index not already loaded
            if market_key not in st.session_state.loaded_indices:
                # Fetch market data for this ticker
                data = fetch_market_data(
                    market_ticker,
                    st.session_state.date_range[0], 
                    st.session_state.date_range[1]
                )
                
                if data is not None and not data.empty:
                    # Calculate technical indicators
                    data = calculate_technical_indicators(data)
                    
                    # Detect drop events
                    drop_events = detect_drop_events(data, st.session_state.drop_threshold)
                    
                    consecutive_drop_events = detect_consecutive_drops(
                        data, 
                        st.session_state.drop_threshold, 
                        st.session_state.consecutive_days
                    ) if st.session_state.consecutive_days > 1 else None
                    
                    # Store data in session state for this specific index
                    st.session_state[f'{market_key}_data'] = data
                    st.session_state[f'{market_key}_drop_events'] = drop_events
                    st.session_state[f'{market_key}_consecutive_drop_events'] = consecutive_drop_events
                    
                    # Keep compatibility with old variable names for S&P 500
                    if market_key == 'sp500':
                        st.session_state.data = data
                        st.session_state.drop_events = drop_events
                        st.session_state.consecutive_drop_events = consecutive_drop_events
                    
                    # Add to loaded indices set
                    st.session_state.loaded_indices.add(market_key)
                    
                    # Success message
                    st.success(f"✅ {market_name} data loaded successfully!")
                    
                elif market_key not in st.session_state.loaded_indices:
                    st.error(f"Failed to fetch {market_name} data. Please check your internet connection and try again.")
                    
            # Make sure current data is set for analysis functions
            if market_key in st.session_state.loaded_indices:
                # Set the active data for compatibility with the analysis functions
                st.session_state.data = st.session_state[f'{market_key}_data']
                st.session_state.drop_events = st.session_state[f'{market_key}_drop_events']
                st.session_state.consecutive_drop_events = st.session_state[f'{market_key}_consecutive_drop_events']
            
        # Populate sub-tabs with content
        with sub_tabs[0]:
            show_historical_performance()

        with sub_tabs[1]:
            show_ml_predictions()

# Add footer
st.markdown("""
<div style="margin-top: 30px; text-align: center; padding: 10px; font-size: 0.8em; color: #6c757d; border-top: 1px solid #f0f2f6;">
    <p style="margin: 5px 0;">
        S&P 500 Market Drop Analyzer - A comprehensive tool for analyzing market correction patterns
    </p>
</div>
""", unsafe_allow_html=True)


