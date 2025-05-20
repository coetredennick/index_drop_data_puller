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
    # Set a default date range - 20 years back from today to today
    twenty_years_ago = (datetime.today() - timedelta(days=365*20)).strftime('%Y-%m-%d')
    st.session_state.date_range = (twenty_years_ago, datetime.today().strftime('%Y-%m-%d'))
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
    
    # Reset data and events when settings change
    st.session_state.data = None
    st.session_state.drop_events = None
    st.session_state.consecutive_drop_events = None
    st.session_state.selected_event = None
    
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

# Main content
# Fetch and process data
with st.spinner("Fetching S&P 500 data..."):
    # Add API settings option
    with st.expander("Data Fetch Settings (Advanced)"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Allow users to enter their own API key for higher rate limits
            use_api_key = st.checkbox("Use Your Yahoo Finance API Key", value=False, 
                                 help="Enable to use your own API key for higher rate limits")
            
            if use_api_key:
                yf_api_key = st.text_input("Yahoo Finance API Key", type="password",
                                      help="Enter your Yahoo Finance API key for higher rate limits")
                if yf_api_key:
                    st.success("API key entered!")
                    # Store the API key in session state for use in data fetching
                    st.session_state.yf_api_key = yf_api_key
            
            # Option to use demo data when Yahoo Finance is unavailable
            if 'use_demo_data' not in st.session_state:
                st.session_state.use_demo_data = False
                
            use_demo_data = st.checkbox("Use Demo Data", value=st.session_state.use_demo_data,
                                   help="Use demonstration data when Yahoo Finance is unavailable due to rate limits")
            
            # Store the demo data preference in session state
            if use_demo_data != st.session_state.use_demo_data:
                st.session_state.use_demo_data = use_demo_data
            
            # Shorter time range for reduced API load
            shorter_range = st.checkbox("Use Shorter Time Range", value=True, 
                                   help="Recommended when facing API rate limits")
        
        with col2:
            retries = st.slider("Max Retries", min_value=1, max_value=10, value=5,
                           help="Number of retry attempts for API calls")
            
            delay = st.slider("Retry Delay (seconds)", min_value=1, max_value=10, value=2,
                         help="Delay between retry attempts")
    
    # Check if we need to reload data based on settings changes
    reload_data = False
    if st.session_state.data is None:
        reload_data = True
    elif not all(date in st.session_state.data.index for date in [st.session_state.date_range[0], st.session_state.date_range[1]]):
        reload_data = True
    
    if reload_data:
        # If shorter range is enabled, adjust the date range to reduce API load
        if shorter_range:
            # Calculate a shorter date range (last 1 year or user selection, whichever is shorter)
            end_date_dt = pd.to_datetime(st.session_state.date_range[1])
            one_year_ago = (end_date_dt - timedelta(days=365)).strftime('%Y-%m-%d')
            adjusted_start_date = max(one_year_ago, st.session_state.date_range[0])
            
            # Notify user of adjustment
            if adjusted_start_date != st.session_state.date_range[0]:
                st.info(f"Date range adjusted to 1 year to reduce API load: {adjusted_start_date} to {st.session_state.date_range[1]}")
                fetch_start_date = adjusted_start_date
            else:
                fetch_start_date = st.session_state.date_range[0]
        else:
            fetch_start_date = st.session_state.date_range[0]
        
        # Set a much smaller date range for initial testing
        today = datetime.today()
        recent_start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # For initial loading, use a very recent, small dataset to avoid rate limits
        if st.session_state.data is None and shorter_range:
            st.info("Loading a recent 30-day sample first to initialize the app. You can load more data after this.")
            fetch_start_date = recent_start_date
        
        # Check if user provided an API key
        api_key = st.session_state.get('yf_api_key', None)
        
        # Attempt to fetch data from Yahoo Finance
        data = None
        
        # First, try the standard method with yfinance
        try:
            data = fetch_sp500_data(
                fetch_start_date, 
                st.session_state.date_range[1],
                max_retries=retries,
                retry_delay=delay,
                api_key=api_key
            )
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        
        # If standard method fails, try pandas-datareader
        if data is None or data.empty:
            st.warning("Trying alternative data source for S&P 500 historical data...")
            
            # Ask if user wants to use pandas-datareader alternative
            if st.button("Fetch Data Using Alternative Source"):
                with st.spinner("Fetching data from alternative source..."):
                    try:
                        # Import the alternative data source module
                        from utils.alternative_data_source import fetch_sp500_alternative
                        
                        # Fetch data using pandas-datareader
                        data = fetch_sp500_alternative(
                            fetch_start_date,
                            st.session_state.date_range[1],
                            max_retries=retries,
                            retry_delay=delay
                        )
                        
                        if data is not None and not data.empty:
                            st.success("Successfully retrieved data from alternative source!")
                    except Exception as alt_error:
                        st.error(f"Error fetching data from alternative source: {alt_error}")
        
        # If API rate limit error occurred
        if data is None:
            # Show friendly guidance
            st.warning("""
            We're having trouble accessing market data from Yahoo Finance. This could be due to:
            
            1. API rate limits (very common)
            2. Network connectivity issues
            3. Data service availability
            
            **What you can do:**
            - Try a smaller date range (e.g., last 3 months instead of multiple years)
            - Wait a few minutes and try again
            - Use the refresh button below
            """)
            
            # Important - give user clear action steps
            if st.button("⟳ Refresh Data (Try Again)"):
                # Clear the cached API failure flag
                if 'yahoo_finance_api_issues' in st.session_state:
                    del st.session_state.yahoo_finance_api_issues
                # Try a smaller date range automatically
                if shorter_range:
                    st.session_state.last_fetch_attempt = recent_start_date
                st.session_state.data = None
                st.rerun()
        
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
            st.error("Failed to fetch S&P 500 data due to API rate limits. Try again later or adjust the date range to a shorter period.")
            
            # Tips for API rate limit issues
            with st.expander("Tips for handling API rate limits"):
                st.markdown("""
                **How to handle Yahoo Finance API rate limits:**
                
                1. **Try shorter time ranges** - Reduce the date range to 1 year or less
                2. **Wait a few minutes** - The rate limits reset after a period of time
                3. **Try again during off-peak hours** - API access may be more available
                4. **Adjust the retry settings** - Increase delay between retry attempts
                """)
                
                st.warning("Remember: This app relies on Yahoo Finance for real market data. Rate limits are a normal part of using their free API service.")

# Create tabs with icons for better visual organization
tabs = st.tabs([
    "📈 Historical Performance", 
    "🤖 ML Predictions"
])

# Add a light separator before tabs content
st.markdown('<hr style="margin-top: 0; margin-bottom: 15px; border: none; height: 1px; background-color: #f0f2f6;">', unsafe_allow_html=True)

# Populate tabs with content
with tabs[0]:
    show_historical_performance()

with tabs[1]:
    show_ml_predictions()

# Add footer
st.markdown("""
<div style="margin-top: 30px; text-align: center; padding: 10px; font-size: 0.8em; color: #6c757d; border-top: 1px solid #f0f2f6;">
    <p style="margin: 5px 0;">
        S&P 500 Market Drop Analyzer - A comprehensive tool for analyzing market correction patterns
    </p>
</div>
""", unsafe_allow_html=True)


