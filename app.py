import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import multiprocessing

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.data_fetcher import fetch_index_data, cache_data
from utils.technical_indicators import calculate_technical_indicators
from utils.event_detection import detect_drop_events, detect_consecutive_drops
from pages.historical_performance import show_historical_performance
from pages.ml_predictions_new import show_ml_predictions

def run_app():
    SUPPORTED_INDICES = {
        "^GSPC": {"name": "S&P 500", "icon": "📈"},
        "^IXIC": {"name": "NASDAQ Composite", "icon": "💻"},
        "^DJI": {"name": "Dow Jones Industrial Average", "icon": "🏦"},
        "^RUT": {"name": "Russell 2000", "icon": "📊"}
    }

    # Configure the page
    st.set_page_config(
        page_title="Market Drop Analyzer",
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
        <h1 style="margin: 0; padding: 0; color: #1E4A7B;">Market Drop Analyzer</h1>
        <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #5A6570;">
            A data-driven tool for analyzing market corrections and forecasting recovery patterns
        </p>
    </div>
    """
    , unsafe_allow_html=True)

    # Index selection
    index_options = list(SUPPORTED_INDICES.keys())
    index_names = [SUPPORTED_INDICES[ticker]['name'] for ticker in index_options]
    
    selected_index_name = st.radio(
        label="Select Market Index:", 
        options=index_names, 
        index=index_options.index(st.session_state.get('active_ticker', '^GSPC')), # Get current or default
        horizontal=True,
        key='selected_index_radio' # Add a key for stability
    )
    
    # Get the ticker symbol corresponding to the selected name
    selected_ticker = index_options[index_names.index(selected_index_name)]

    # Update active_ticker in session state if it has changed
    if st.session_state.get('active_ticker') != selected_ticker:
        st.session_state.active_ticker = selected_ticker
        # Ensure new ticker has default entries initialized BEFORE rerun
        if selected_ticker not in st.session_state.market_data_store:
            st.session_state.market_data_store[selected_ticker] = None
        if selected_ticker not in st.session_state.drop_events_store:
            st.session_state.drop_events_store[selected_ticker] = None
        if selected_ticker not in st.session_state.consecutive_drop_events_store:
            st.session_state.consecutive_drop_events_store[selected_ticker] = None
        if selected_ticker not in st.session_state.selected_event_store:
            st.session_state.selected_event_store[selected_ticker] = None
        if selected_ticker not in st.session_state.current_event_type_filter_store:
            st.session_state.current_event_type_filter_store[selected_ticker] = 'all'
        if selected_ticker not in st.session_state.ml_models_store or st.session_state.ml_models_store[selected_ticker] is None:
            st.session_state.ml_models_store[selected_ticker] = {horizon: None for horizon in ["1W", "1M", "3M", "6M", "1Y"]}
        if selected_ticker not in st.session_state.ml_model_params_store or st.session_state.ml_model_params_store[selected_ticker] is None:
            st.session_state.ml_model_params_store[selected_ticker] = {horizon: {} for horizon in ["1W", "1M", "3M", "6M", "1Y"]}
        st.rerun() # Rerun the app to reflect the new index selection

    # Initialize session state for settings
    if 'drop_threshold' not in st.session_state:
        st.session_state.drop_threshold = 0.1
    if 'consecutive_days' not in st.session_state:
        st.session_state.consecutive_days = 1
    if 'date_range' not in st.session_state:
        st.session_state.date_range = ('1990-01-01', datetime.today().strftime('%Y-%m-%d'))
    
    # Initialize dictionaries to store data for each ticker if they don't exist
    # These will hold the data for each index (e.g., market_data_store['^GSPC'] = S&P data)
    if 'market_data_store' not in st.session_state:
        st.session_state.market_data_store = {}
    if 'drop_events_store' not in st.session_state:
        st.session_state.drop_events_store = {}
    if 'consecutive_drop_events_store' not in st.session_state:
        st.session_state.consecutive_drop_events_store = {}
    if 'selected_event_store' not in st.session_state:
        st.session_state.selected_event_store = {}
    if 'current_event_type_filter_store' not in st.session_state:
        st.session_state.current_event_type_filter_store = {}
    if 'ml_models_store' not in st.session_state:
        st.session_state.ml_models_store = {}
    if 'ml_model_params_store' not in st.session_state:
        st.session_state.ml_model_params_store = {}

    # Ensure the current active_ticker has default entries in these new stores.
    # This replaces the old single-value session state initializations like 'if "data" not in st.session_state: st.session_state.data = None'
    active_ticker = st.session_state.active_ticker # Get the active ticker once

    if active_ticker not in st.session_state.market_data_store:
        st.session_state.market_data_store[active_ticker] = None
    if active_ticker not in st.session_state.drop_events_store:
        st.session_state.drop_events_store[active_ticker] = None
    if active_ticker not in st.session_state.consecutive_drop_events_store:
        st.session_state.consecutive_drop_events_store[active_ticker] = None
    if active_ticker not in st.session_state.selected_event_store:
        st.session_state.selected_event_store[active_ticker] = None
    if active_ticker not in st.session_state.current_event_type_filter_store:
        st.session_state.current_event_type_filter_store[active_ticker] = 'all'
    if active_ticker not in st.session_state.ml_models_store or st.session_state.ml_models_store[active_ticker] is None:
        st.session_state.ml_models_store[active_ticker] = {horizon: None for horizon in ["1W", "1M", "3M", "6M", "1Y"]}
    if active_ticker not in st.session_state.ml_model_params_store or st.session_state.ml_model_params_store[active_ticker] is None:
        st.session_state.ml_model_params_store[active_ticker] = {horizon: {} for horizon in ["1W", "1M", "3M", "6M", "1Y"]}

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
        st.session_state.market_data_store[active_ticker] = None
        st.session_state.drop_events_store[active_ticker] = None
        st.session_state.consecutive_drop_events_store[active_ticker] = None
        st.session_state.selected_event_store[active_ticker] = None
        
        # Show info message
        st.success("✅ Settings applied! Data will be refreshed.")
        st.rerun()

    # Data source info with improved styling
    st.markdown("""
    <div style="text-align: right; font-size: 0.8em; color: #5A6570; margin-top: -0.5rem; margin-bottom: 0.7rem;">
        <span style="background-color: #f0f2f6; padding: 0.2rem 0.5rem; border-radius: 3px;">
            <i>Data source: Yahoo Finance ({})</i>
        </span>
    </div>
    """.format(active_ticker), unsafe_allow_html=True)

    # Main content
    # Fetch and process data
    with st.spinner("Fetching {} data...".format(SUPPORTED_INDICES[active_ticker]["name"])):
        # Check if we need to reload data based on settings changes
        reload_data = False
        if st.session_state.market_data_store[active_ticker] is None:
            reload_data = True
        elif not all(date in st.session_state.market_data_store[active_ticker].index for date in [st.session_state.date_range[0], st.session_state.date_range[1]]):
            reload_data = True
        
        if reload_data:
            # Fetch data
            data = fetch_index_data(
                active_ticker, 
                SUPPORTED_INDICES[active_ticker]['name'], 
                st.session_state.date_range[0], 
                st.session_state.date_range[1]
            )
            
            if data is not None and not data.empty:
                # Calculate technical indicators
                data = calculate_technical_indicators(data)
                print(f"DEBUG: Columns in data for {SUPPORTED_INDICES[active_ticker]['name']} after tech indicators: {data.columns.tolist()}")
                if 'ATRr_14' in data.columns:
                    print(f"DEBUG: ATRr_14 head for {SUPPORTED_INDICES[active_ticker]['name']}:\n{data['ATRr_14'].head()}")
                    print(f"DEBUG: ATRr_14 tail for {SUPPORTED_INDICES[active_ticker]['name']}:\n{data['ATRr_14'].tail()}")
                else:
                    print(f"DEBUG: ATRr_14 column MISSING for {SUPPORTED_INDICES[active_ticker]['name']}")
                if 'ATR_Pct' in data.columns:
                    print(f"DEBUG: ATR_Pct head for {SUPPORTED_INDICES[active_ticker]['name']}:\n{data['ATR_Pct'].head()}")
                    print(f"DEBUG: ATR_Pct tail for {SUPPORTED_INDICES[active_ticker]['name']}:\n{data['ATR_Pct'].tail()}")
                else:
                    print(f"DEBUG: ATR_Pct column MISSING for {SUPPORTED_INDICES[active_ticker]['name']}")

                # Detect drop events
                drop_events = detect_drop_events(data, st.session_state.drop_threshold)
                
                consecutive_drop_events = detect_consecutive_drops(
                    data, 
                    st.session_state.drop_threshold, 
                    st.session_state.consecutive_days
                ) if st.session_state.consecutive_days > 1 else None
                
                # Update session state
                st.session_state.market_data_store[active_ticker] = data
                st.session_state.drop_events_store[active_ticker] = drop_events
                st.session_state.consecutive_drop_events_store[active_ticker] = consecutive_drop_events
                
                # Cache data
                cache_data(data, drop_events, consecutive_drop_events)
            else:
                st.error("Failed to fetch {} data. Please check your internet connection and try again.".format(SUPPORTED_INDICES[active_ticker]["name"]))

    # Create tabs with icons for better visual organization
    tabs = st.tabs([
        "{} Historical Performance".format(SUPPORTED_INDICES[active_ticker]["icon"]), 
        "{} ML Predictions".format(SUPPORTED_INDICES[active_ticker]["icon"])
    ])

    # Add a light separator before tabs content
    st.markdown('<hr style="margin-top: 0; margin-bottom: 15px; border: none; height: 1px; background-color: #f0f2f6;">', unsafe_allow_html=True)

    # Populate tabs with content
    with tabs[0]:
        if st.session_state.market_data_store.get(active_ticker) is not None:
            st.markdown(f"### {SUPPORTED_INDICES[active_ticker]['icon']} Historical Performance for {SUPPORTED_INDICES[active_ticker]['name']}")
            show_historical_performance(
                market_data=st.session_state.market_data_store[active_ticker],
                drop_events_for_ticker=st.session_state.drop_events_store[active_ticker],
                consecutive_drop_events_for_ticker=st.session_state.consecutive_drop_events_store[active_ticker],
                active_ticker_symbol=active_ticker,
                active_ticker_name=SUPPORTED_INDICES[active_ticker]['name']
            )

    with tabs[1]:
        # Ensure data for the ML predictions tab is also available
        if st.session_state.market_data_store.get(active_ticker) is not None:
            st.markdown(f"### {SUPPORTED_INDICES[active_ticker]['icon']} ML Predictions for {SUPPORTED_INDICES[active_ticker]['name']}")
            show_ml_predictions(
                market_data=st.session_state.market_data_store[active_ticker],
                ml_models_for_ticker=st.session_state.ml_models_store[active_ticker],
                ml_model_params_for_ticker=st.session_state.ml_model_params_store[active_ticker],
                active_ticker_symbol=active_ticker,
                active_ticker_name=SUPPORTED_INDICES[active_ticker]['name']
            )
        else:
            st.warning(f"Data for {SUPPORTED_INDICES[active_ticker]['name']} is not yet loaded. Please wait or check data fetching settings.")

    # Add footer
    st.markdown("""
    <div style="margin-top: 30px; text-align: center; padding: 10px; font-size: 0.8em; color: #6c757d; border-top: 1px solid #f0f2f6;">
        <p style="margin: 5px 0;">
            Market Drop Analyzer - A comprehensive tool for analyzing market correction patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_app()
