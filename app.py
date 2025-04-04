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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to create a modern, professional financial dashboard
st.markdown("""
<style>
    /* Main container adjustments */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Improved sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f5f7fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Header and title styling */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #1E3A8A;
    }
    h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    h2 {
        font-size: 1.8rem;
        margin-top: 1.5rem;
    }
    h3 {
        font-size: 1.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        color: #2E4057;
        border-left: 4px solid #1E88E5;
        padding-left: 0.5rem;
    }
    
    /* Improved tab navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f5f7fa;
        padding: 0.5rem 0.5rem 0 0.5rem;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 136, 229, 0.05);
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid #e9ecef;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #1E88E5;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 136, 229, 0.1);
    }
    
    /* Card styling for consistent look */
    .card {
        border-radius: 10px;
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Metric widgets styling */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    div[data-testid="stMetric"] > div:first-child {
        color: #1E3A8A;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Improve spacing */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) {
        margin-bottom: 0.5rem;
    }
    
    /* Slider and select inputs */
    div[data-testid="stSlider"] {
        padding: 1rem 0;
    }
    
    /* Legend styling for charts */
    .js-plotly-plot .legend {
        font-family: sans-serif;
        font-size: 12px;
    }
    
    /* Add subtle dividers between sections */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Create a more engaging header with icon and title
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <div style="font-size: 3rem; margin-right: 0.8rem; color: #1E88E5;">📊</div>
    <div>
        <h1 style="margin: 0; padding: 0; color: #1E3A8A;">S&P 500 Market Drop Analyzer</h1>
        <p style="margin: 0; padding: 0; color: #6B7280; font-size: 1.1rem;">Analyze market corrections with data-driven insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Information card with app description
st.markdown("""
<div style="background-color: white; border-radius: 10px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
    <div style="display: flex; align-items: flex-start;">
        <div style="flex: 1;">
            <p style="margin-top: 0;">
                This dashboard provides comprehensive analysis of significant market corrections in the S&P 500 index.
                It combines historical data analysis, technical indicators, interactive visualizations, and machine learning
                to help understand market behavior during and after significant drop events.
            </p>
            <div style="display: flex; margin-top: 15px;">
                <div style="margin-right: 20px; display: flex; align-items: center;">
                    <span style="color: #1E88E5; font-size: 1.2rem; margin-right: 5px;">📈</span>
                    <span>Track recovery patterns</span>
                </div>
                <div style="margin-right: 20px; display: flex; align-items: center;">
                    <span style="color: #1E88E5; font-size: 1.2rem; margin-right: 5px;">🔍</span>
                    <span>Analyze technical indicators</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="color: #1E88E5; font-size: 1.2rem; margin-right: 5px;">🤖</span>
                    <span>ML-powered predictions</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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

# Sidebar for settings
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #1E3A8A; margin-bottom: 5px;">⚙️ Analysis Settings</h2>
        <p style="font-size: 0.9rem; color: #6B7280; margin-top: 0;">Customize your analysis parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a card-like container for date range selection
    st.markdown("""
    <div style="background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <h3 style="margin-top: 0; color: #1E88E5; font-size: 1.2rem; border-bottom: 1px solid #e9ecef; padding-bottom: 8px;">
            📅 Data Range
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Date Range Selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=pd.to_datetime(st.session_state.date_range[0]),
            min_value=pd.to_datetime('1950-01-01'),
            max_value=datetime.today() - timedelta(days=1)
        )
    with col2:
        end_date = st.date_input(
            "To",
            value=pd.to_datetime(st.session_state.date_range[1]),
            min_value=pd.to_datetime('1950-01-01'),
            max_value=datetime.today()
        )
    
    # Create a card-like container for drop event detection
    st.markdown("""
    <div style="background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <h3 style="margin-top: 0; color: #1E88E5; font-size: 1.2rem; border-bottom: 1px solid #e9ecef; padding-bottom: 8px;">
            📉 Drop Event Detection
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Drop Events Detection Settings
    drop_threshold = st.slider(
        "Drop Threshold (%)",
        min_value=0.1,
        max_value=20.0,
        value=st.session_state.drop_threshold,
        step=0.1,
        help="Minimum percentage drop to be considered as a significant market event"
    )
    
    # Add some information about the threshold selection
    threshold_info = ""
    if drop_threshold < 1.0:
        threshold_info = "Very sensitive: Will detect minor fluctuations"
    elif drop_threshold < 3.0:
        threshold_info = "Moderate: Will detect regular corrections"
    elif drop_threshold < 7.0:
        threshold_info = "High: Will detect significant corrections"
    else:
        threshold_info = "Extreme: Will only detect major market crashes"
    
    st.markdown(f"""
    <div style="font-size: 0.8rem; margin-top: -15px; margin-bottom: 15px; color: #6B7280;">
        {threshold_info}
    </div>
    """, unsafe_allow_html=True)
    
    # Consecutive drops detection with improved styling
    use_consecutive = st.checkbox(
        "Detect Consecutive Day Drops",
        value=st.session_state.consecutive_days > 1,
        help="Detect sequences of consecutive days where each day fell by more than the threshold"
    )
    
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
    
    # Apply button with improved styling
    st.markdown("<div style='padding: 10px 0;'></div>", unsafe_allow_html=True)
    
    if st.button("Apply Settings", use_container_width=True):
        # Update session state
        st.session_state.drop_threshold = drop_threshold
        st.session_state.consecutive_days = consecutive_days if use_consecutive else 1
        st.session_state.date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Reset selected event when settings change
        st.session_state.selected_event = None
        
        # Show info message
        st.success("Settings applied! Data will be refreshed.")
        st.rerun()
    
    # About section with improved styling
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: white; border-radius: 10px; padding: 15px; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <h3 style="margin-top: 0; color: #1E88E5; font-size: 1.2rem; border-bottom: 1px solid #e9ecef; padding-bottom: 8px;">
            ℹ️ About
        </h3>
        <p style="font-size: 0.9rem; margin-bottom: 5px;">
            This tool analyzes S&P 500 market drops to help understand patterns
            and recovery trajectories after significant market corrections.
        </p>
        <div style="font-size: 0.8rem; color: #6B7280; margin-top: 10px;">
            <strong>Data source:</strong> Yahoo Finance (^GSPC)
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content
# Custom loading indicator using HTML/CSS
loading_html = """
<div style="display: flex; justify-content: center; align-items: center; margin: 2rem 0; flex-direction: column;">
    <div style="width: 80px; height: 80px; border: 4px solid #f3f3f3; border-top: 4px solid #1E88E5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
    <p style="margin-top: 1rem; font-size: 1.2rem; color: #2E4057; font-weight: 600;">Loading financial data...</p>
    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #6B7280;">This may take a moment as we calculate technical indicators and identify market events</p>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</div>
"""

# Fetch and process data
# Check if we need to reload data based on settings changes
reload_data = False
if st.session_state.data is None:
    reload_data = True
elif not all(date in st.session_state.data.index for date in [st.session_state.date_range[0], st.session_state.date_range[1]]):
    reload_data = True

if reload_data:
    # Show the custom loading spinner
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
    
    try:
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
            
            # Show a success message
            loading_placeholder.empty()
            st.success(f"Successfully loaded {len(data)} days of market data with {len(drop_events)} drop events detected.")
        else:
            # Show an error message
            loading_placeholder.empty()
            st.error("Failed to fetch S&P 500 data. Please check your internet connection and try again.")
    except Exception as e:
        # Show an error message
        loading_placeholder.empty()
        st.error(f"An error occurred while processing data: {str(e)}")
        st.info("Please try adjusting your date range or reload the application.")

# Create tabs with icons for different sections
tab_labels = [
    "📊 Historical Performance", 
    "📉 Drop Events Analysis", 
    "📈 Current Market", 
    "🤖 ML Predictions"
]

st.markdown("""
<div style="margin-bottom: 0; background-color: #f5f7fa; padding: 10px 15px 0; border-radius: 10px 10px 0 0; border: 1px solid #e9ecef; border-bottom: none;">
    <p style="font-size: 0.85rem; color: #6B7280; margin: 0 0 8px 0;">
        Select an analysis view to explore different aspects of market drops:
    </p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tabs = st.tabs(tab_labels)

# Populate tabs with content
with tabs[0]:
    show_historical_performance()

with tabs[1]:
    show_drop_events()

with tabs[2]:
    show_current_market()

with tabs[3]:
    show_ml_predictions()
