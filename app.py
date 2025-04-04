import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.data_fetcher import fetch_sp500_data, cache_data
from utils.technical_indicators import calculate_technical_indicators
from utils.event_detection import detect_drop_events, detect_consecutive_drops
from pages.historical_performance import show_historical_performance
from pages.drop_events import show_drop_events
from pages.current_market import show_current_market
from pages.ml_predictions import show_ml_predictions

# Configure the page
st.set_page_config(
    page_title="S&P 500 Market Drop Analyzer",
    page_icon="📉",
    layout="wide"
)

# Add some CSS to make it look like a professional financial dashboard
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Hide the default Streamlit navigation sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide the hamburger menu that shows the navigation sidebar */
    button[kind="header"] {
        display: none !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #0E6EFD;
    }
    .css-1y4p8pa {
        max-width: 1200px;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(1) {
        margin-bottom: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("S&P 500 Market Drop Analyzer")
st.markdown("""
This dashboard provides comprehensive analysis of significant market corrections in the S&P 500 index.
It combines historical data analysis, technical indicators, interactive visualizations, and machine learning
to help understand market behavior during and after significant drop events.
""")

# Initialize session state for settings
if 'drop_threshold' not in st.session_state:
    st.session_state.drop_threshold = 2.0
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

# Main page settings
# Create a container for the analysis settings
st.markdown("### Analysis Settings")

# Use columns to organize the settings in a more compact horizontal layout
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# Date Range Selection in the first column
with col1:
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime(st.session_state.date_range[0]),
        min_value=pd.to_datetime('1950-01-01'),
        max_value=datetime.today() - timedelta(days=1)
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime(st.session_state.date_range[1]),
        min_value=pd.to_datetime('1950-01-01'),
        max_value=datetime.today()
    )

# Drop Events Detection Settings in the third column
with col3:
    drop_threshold = st.slider(
        "Drop Threshold (%)",
        min_value=0.1,
        max_value=20.0,
        value=st.session_state.drop_threshold,
        step=0.1,
        help="Minimum percentage drop to be considered as a significant market event"
    )
    
    use_consecutive = st.checkbox(
        "Detect Consecutive Day Drops",
        value=st.session_state.consecutive_days > 1,
        help="Detect sequences of consecutive days where each day fell by more than the threshold"
    )

# Fourth column for consecutive days and apply button
with col4:
    consecutive_days = 1
    if use_consecutive:
        consecutive_days = st.slider(
            "Number of Consecutive Days",
            min_value=2,
            max_value=5,
            value=max(2, st.session_state.consecutive_days),
            step=1,
            help="Number of consecutive days each with drops exceeding the threshold"
        )
    
    # Apply button
    if st.button("Apply Settings", key="apply_settings"):
        # Update session state
        st.session_state.drop_threshold = drop_threshold
        st.session_state.consecutive_days = consecutive_days if use_consecutive else 1
        st.session_state.date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Reset selected event when settings change
        st.session_state.selected_event = None
        
        # Show info message
        st.info("Settings applied! Data will be refreshed.")
        st.rerun()

# Data source info
st.markdown("""
<div style="text-align: right; font-size: 0.8em; color: gray;">
Data source: Yahoo Finance (^GSPC)
</div>
""", unsafe_allow_html=True)

# Main content
# Fetch and process data
with st.spinner("Fetching S&P 500 data..."):
    # Check if we need to reload data based on settings changes
    reload_data = False
    if st.session_state.data is None:
        reload_data = True
    elif not all(date in st.session_state.data.index for date in [st.session_state.date_range[0], st.session_state.date_range[1]]):
        reload_data = True
    
    if reload_data:
        # Fetch data
        data = fetch_sp500_data(st.session_state.date_range[0], st.session_state.date_range[1])
        
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
            
            # Update session state
            st.session_state.data = data
            st.session_state.drop_events = drop_events
            st.session_state.consecutive_drop_events = consecutive_drop_events
            
            # Cache data
            cache_data(data, drop_events, consecutive_drop_events)
        else:
            st.error("Failed to fetch S&P 500 data. Please check your internet connection and try again.")

# Create tabs for different sections
tabs = st.tabs([
    "Historical Performance", 
    "Drop Events Analysis", 
    "Current Market Conditions", 
    "ML Predictions"
])

# Populate tabs with content
with tabs[0]:
    show_historical_performance()

with tabs[1]:
    show_drop_events()

with tabs[2]:
    show_current_market()

with tabs[3]:
    show_ml_predictions()
