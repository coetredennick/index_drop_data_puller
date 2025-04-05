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
from utils.unified_visualization import create_unified_visualization_ui, get_ml_prediction_data
from utils.ml_models import prepare_features, train_model

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
        max-width: 1300px;
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

    /* Button styling */
    .stButton > button {
        background-color: #1E88E5;
        color: white !important;
        border-radius: 4px;
        border: none;
        padding: 0.4rem 0.8rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #1976D2;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Active button styling */
    .stButton > button.active {
        background-color: #0D47A1;
        color: white;
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
    
    /* Card styling for metrics */
    .metric-card {
        background-color: white;
        border: 1px solid #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Checkbox styling */
    .stCheckbox > div > div > label {
        display: flex;
        align-items: center;
    }
    
    /* Tabs styling for the new design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 3px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #f8f9fa;
        border-radius: 4px;
        padding: 10px 16px;
        font-size: 14px;
        font-weight: 500;
        color: #5A6570;
        border: none;
        margin: 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Improved data table styling */
    div.stDataFrame {
        font-size: 13px;
    }
    
    div.stDataFrame td {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    div.stDataFrame th {
        font-weight: 600;
        background-color: #f0f2f6;
    }
    
    /* Fix the chart height */
    .js-plotly-plot {
        min-height: 400px;
    }
</style>
""", unsafe_allow_html=True)

# Cleaner title and description layout
st.markdown("""
<div style="text-align: center; padding: 1rem 0; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 1rem;">
    <h1 style="margin: 0; padding: 0; color: #1E4A7B;">S&P 500 Market Drop Analyzer</h1>
    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #5A6570;">
        Simplified visualization of market performance with drop event analysis and ML forecasting
    </p>
</div>
"""
, unsafe_allow_html=True)

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
if 'ml_model_result' not in st.session_state:
    st.session_state.ml_model_result = None
if 'time_period' not in st.session_state:
    st.session_state.time_period = "All Data"

# Main page settings in a clean, collapsible container
with st.expander("📊 Analysis Settings", expanded=False):
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
                step=0.1,
                help="Minimum percentage drop to be considered as a significant market event"
            )
            
            detection_col1, detection_col2 = st.columns([3, 2])
            
            with detection_col1:
                use_consecutive = st.checkbox(
                    "Detect Consecutive Drops",
                    value=st.session_state.consecutive_days > 1,
                    help="Detect sequences of consecutive days where each day fell by more than the threshold"
                )
            
            with detection_col2:
                consecutive_days = 1
                if use_consecutive:
                    consecutive_days = st.number_input(
                        "Days",
                        min_value=2,
                        max_value=5,
                        value=max(2, st.session_state.consecutive_days),
                        help="Number of consecutive days each with drops exceeding the threshold"
                    )
        
        # Form submit button
        submit_button = st.form_submit_button("Apply Settings", use_container_width=True)
    
    # Process form submission outside the form block
    if submit_button:
        # Update session state
        st.session_state.drop_threshold = drop_threshold
        st.session_state.consecutive_days = consecutive_days if use_consecutive else 1
        st.session_state.date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Reset selected event when settings change
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

# Create tabs with simplified organization
tab1, tab2, tab3 = st.tabs([
    "📈 Market Visualization", 
    "🔍 Market Analysis", 
    "🤖 ML Predictions"
])

# Main visualization tab
with tab1:
    create_unified_visualization_ui(
        data=st.session_state.data, 
        drop_events=st.session_state.drop_events, 
        consecutive_drop_events=st.session_state.consecutive_drop_events,
        ml_model_result=st.session_state.ml_model_result
    )

# Market Analysis Tab
with tab2:
    # Key market metrics
    st.write("## Market Drop Event Analysis")
    
    # Show analysis metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Count drop events
    total_drops = len(st.session_state.drop_events or [])
    consecutive_drops = len(st.session_state.consecutive_drop_events or [])
    
    # Calculate average drop size
    avg_drop = 0
    if st.session_state.drop_events and len(st.session_state.drop_events) > 0:
        avg_drop = sum(event['drop'] for event in st.session_state.drop_events) / len(st.session_state.drop_events)
    
    # Calculate largest drop
    largest_drop = 0
    if st.session_state.drop_events and len(st.session_state.drop_events) > 0:
        largest_drop = max(event['drop'] for event in st.session_state.drop_events)
    
    with metrics_col1:
        st.metric("Single Day Drop Events", total_drops)
    
    with metrics_col2:
        st.metric("Consecutive Drop Events", consecutive_drops)
    
    with metrics_col3:
        st.metric("Average Drop Size", f"{avg_drop:.2f}%")
    
    with metrics_col4:
        st.metric("Largest Single Day Drop", f"{largest_drop:.2f}%")
    
    # Create a table of drop events
    st.write("### Drop Event Database")
    
    # Combine all events
    all_events = []
    if st.session_state.drop_events:
        all_events.extend(st.session_state.drop_events)
    if st.session_state.consecutive_drop_events:
        all_events.extend(st.session_state.consecutive_drop_events)
    
    # Sort events by date
    all_events.sort(key=lambda x: x['date'] if 'date' in x else x['start_date'], reverse=True)
    
    if not all_events:
        st.info("No significant drop events detected with current settings. Try adjusting the drop threshold.")
    else:
        # Create a dataframe for the events
        events_data = []
        for event in all_events:
            if 'consecutive' in event and event['consecutive']:
                events_data.append({
                    'Event Type': 'Consecutive',
                    'Date': f"{event['start_date']} to {event['end_date']}",
                    'Drop (%)': f"{event['total_drop']:.2f}%",
                    'Days': event['days'],
                    'Severity': event.get('severity', 'N/A')
                })
            else:
                events_data.append({
                    'Event Type': 'Single Day',
                    'Date': event['date'],
                    'Drop (%)': f"{event['drop']:.2f}%",
                    'Days': 1,
                    'Severity': event.get('severity', 'N/A')
                })
        
        events_df = pd.DataFrame(events_data)
        st.dataframe(events_df, use_container_width=True)
    
    # Recovery analysis
    st.write("### Aggregate Recovery Analysis")
    
    recovery_col1, recovery_col2 = st.columns(2)
    
    with recovery_col1:
        st.write("#### Average Returns After Drop Events")
        
        # Calculate average returns for different periods
        if all_events and len(all_events) > 0 and st.session_state.data is not None:
            # Periods to analyze
            periods = ['1W', '1M', '3M', '6M', '1Y']
            avg_returns = []
            
            for period in periods:
                # Get column name for this period
                col_name = f'Fwd_Ret_{period}'
                
                # Calculate the forward returns for each event
                returns = []
                for event in all_events:
                    event_date = pd.to_datetime(event['date'] if 'date' in event else event['end_date'])
                    if event_date in st.session_state.data.index and col_name in st.session_state.data.columns:
                        ret = st.session_state.data.loc[event_date, col_name]
                        if pd.notna(ret):
                            returns.append(ret)
                
                # Calculate average return for this period
                if returns:
                    avg_return = sum(returns) / len(returns)
                    avg_returns.append({
                        'Period': period,
                        'Average Return (%)': f"{avg_return:.2f}%",
                        'Sample Size': len(returns)
                    })
                else:
                    avg_returns.append({
                        'Period': period,
                        'Average Return (%)': 'N/A',
                        'Sample Size': 0
                    })
            
            # Create dataframe for average returns
            avg_returns_df = pd.DataFrame(avg_returns)
            st.dataframe(avg_returns_df, use_container_width=True)
        else:
            st.info("No drop events detected or data not available for recovery analysis.")
    
    with recovery_col2:
        st.write("#### Market Drop Seasonality")
        
        # Analyze monthly distribution of drops
        if all_events and len(all_events) > 0:
            # Count drops by month
            monthly_drops = [0] * 12
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for event in all_events:
                event_date = pd.to_datetime(event['date'] if 'date' in event else event['start_date'])
                month_idx = event_date.month - 1
                monthly_drops[month_idx] += 1
            
            # Create dataframe for monthly distribution
            monthly_df = pd.DataFrame({
                'Month': month_names,
                'Drop Count': monthly_drops
            })
            
            # Sort by drop count
            monthly_df = monthly_df.sort_values('Drop Count', ascending=False)
            
            st.dataframe(monthly_df, use_container_width=True)
        else:
            st.info("No drop events detected for seasonality analysis.")

# ML Predictions Tab
with tab3:
    st.write("## Machine Learning Price Forecasting")
    
    # ML model settings
    with st.expander("🤖 ML Model Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                options=["random_forest", "gradient_boosting", "linear_regression"],
                index=0,
                help="Type of machine learning model to use for predictions"
            )
        
        with col2:
            target_period = st.selectbox(
                "Target Period",
                options=["1W", "1M", "3M", "6M", "1Y"],
                index=1,
                help="Time period for which to predict returns"
            )
        
        with col3:
            focus_on_drops = st.checkbox(
                "Focus on Market Drops",
                value=True,
                help="Train the model specifically on market drop events for better prediction"
            )
    
    # Train model button
    train_model_btn = st.button("Train ML Model", use_container_width=True)
    
    if train_model_btn:
        with st.spinner("Training machine learning model..."):
            if st.session_state.data is not None and not st.session_state.data.empty:
                # Prepare features for ML model
                data_with_features, features = prepare_features(
                    st.session_state.data,
                    focus_on_drops=focus_on_drops,
                    drop_threshold=-st.session_state.drop_threshold
                )
                
                # Train the model
                target_column = f'Fwd_Ret_{target_period}'
                ml_result = train_model(
                    data_with_features, 
                    features, 
                    target_column, 
                    model_type=model_type
                )
                
                if ml_result['success']:
                    # Format ML prediction data for unified chart
                    prediction_data = get_ml_prediction_data(
                        ml_result,
                        st.session_state.data,
                        features,
                        days_to_forecast=30
                    )
                    
                    # Store ML model result in session state
                    st.session_state.ml_model_result = prediction_data
                    st.success("✅ ML model trained successfully! View forecast in the Market Visualization tab.")
                else:
                    st.error(f"Failed to train ML model: {ml_result.get('error', 'Unknown error')}")
            else:
                st.error("No data available for ML training. Please check data settings.")
    
    # Display ML prediction results
    if st.session_state.ml_model_result is not None and st.session_state.ml_model_result.get('success', False):
        st.write("### ML Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Target Period")
            st.info(f"The model is trained to predict returns over {target_period} time horizon.")
            
            # Display prediction details
            st.write("#### Latest S&P 500 Price")
            latest_price = 0
            if st.session_state.data is not None and not st.session_state.data.empty:
                latest_price = st.session_state.data['Close'].iloc[-1]
                latest_date = st.session_state.data.index[-1]
                st.metric("Latest Price", f"${latest_price:.2f}", delta=None)
                st.caption(f"As of {latest_date.strftime('%Y-%m-%d')}")
            
            # Display forecast
            st.write("#### ML Price Forecast")
            forecast_dates = st.session_state.ml_model_result.get('forecast_dates', [])
            forecast_prices = st.session_state.ml_model_result.get('forecast_prices', [])
            
            if forecast_dates and forecast_prices and latest_price > 0:
                forecast_end_date = forecast_dates[-1]
                forecast_end_price = forecast_prices[-1]
                
                forecast_percent = ((forecast_end_price / latest_price) - 1) * 100
                delta_text = f"{forecast_percent:.2f}%" if forecast_percent >= 0 else f"{forecast_percent:.2f}%"
                
                st.metric(
                    f"Forecast ({forecast_end_date.strftime('%Y-%m-%d')})", 
                    f"${forecast_end_price:.2f}", 
                    delta=delta_text
                )
        
        with col2:
            st.write("#### Forecast Instructions")
            st.markdown("""
            **To view the S&P 500 forecast visualization:**
            1. Go to the **Market Visualization** tab
            2. Check the **Show ML Forecast** checkbox
            3. The forecast will appear as a green line continuing from the blue price line
            4. Lighter green shading shows the confidence interval
            
            The ML model analyzes historical patterns, technical indicators, and market behavior 
            to generate these predictions.
            """)
    else:
        st.info("No ML model has been trained yet. Click the 'Train ML Model' button to create a forecast.")

# Add footer
st.markdown("""
<div style="margin-top: 30px; text-align: center; padding: 10px; font-size: 0.8em; color: #6c757d; border-top: 1px solid #f0f2f6;">
    <p style="margin: 5px 0;">
        S&P 500 Market Drop Analyzer - A simplified visualization tool for market analysis
    </p>
</div>
""", unsafe_allow_html=True)
