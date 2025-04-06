import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.ml_models_new import (
    prepare_features, 
    train_model, 
    predict_returns, 
    create_prediction_chart,
    create_feature_importance_chart,
    create_forecast_chart
)

def show_ml_predictions():
    """
    Display the Machine Learning Predictions tab focused on ML forecasting 
    with confidence intervals for different time periods (1W, 1M, 3M, 1Y)
    Shows a forecasting graph with YTD data and up to 1 year projection
    """
    
    # Add custom styling
    st.markdown("""
    <style>
        /* Card styling for sections */
        div.stMarkdown h3 {
            background-color: #f8f9fa;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            color: #1E4A7B;
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        /* Improve plot styling */
        .stPlotlyChart {
            margin-bottom: 1rem;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1rem;
        }
        
        /* Enhanced forecast chart */
        .forecast-chart {
            margin-top: 1rem;
            margin-bottom: 1.5rem;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Confidence interval cards */
        .confidence-card {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1E88E5;
        }
        
        /* Metric styling */
        div[data-testid="stMetric"] {
            background-color: white;
            padding: 0.7rem;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown("### S&P 500 ML Forecasting")
    
    # Check if data is available
    data_available = st.session_state.data is not None and not st.session_state.data.empty
    
    if not data_available:
        st.warning("No data available. Please adjust the date range and fetch data.")
        return
    
    # Initialize session state for ML models
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = {
            '1W': None, 
            '1M': None, 
            '3M': None,
            '6M': None, 
            '1Y': None
        }
    
    # Current market information
    latest_data = st.session_state.data.iloc[-1]
    latest_date = latest_data.name.strftime('%Y-%m-%d') if hasattr(latest_data.name, 'strftime') else str(latest_data.name)
    latest_close = latest_data['Close']
    latest_return = latest_data['Return'] if 'Return' in latest_data else 0
    
    # Display current market status prominently
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 8px; background-color: #f0f4f8; margin: 15px 0; border: 1px solid #d0e1f9;">
        <h3 style="margin: 0; color: #104E8B; font-size: 1.2rem;">Current S&P 500</h3>
        <div style="display: flex; align-items: center; margin-top: 8px;">
            <div style="font-size: 28px; font-weight: 700; margin-right: 15px;">${latest_close:.2f}</div>
            <div style="font-size: 16px; font-weight: 500; color: {'#21A366' if latest_return >= 0 else '#E63946'};">
                {latest_return:+.2f}% today
            </div>
        </div>
        <div style="font-size: 12px; color: #666; margin-top: 5px;">Last updated: {latest_date}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different model configurations
    st.markdown("#### Forecast Configuration")
    
    # Show the current drop threshold from main settings
    st.markdown(f"""
    <div style="margin-bottom: 15px; padding: 8px 12px; background-color: #f0f8ff; border-left: 3px solid #1E88E5; border-radius: 4px;">
        <p style="margin: 0; font-size: 0.85rem;">
            <strong>Using Drop Threshold:</strong> {st.session_state.drop_threshold:.1f}% 
            <span style="color: #666; font-style: italic;">(synchronized with main settings)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Forecast settings in a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Only use Random Forest model - the best model type for this application
        st.markdown("""
        <div style="padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 4px solid #4682B4;">
            <p style="margin: 0; font-weight: 600; color: #104E8B;">🌲 Using Optimized Random Forest Model</p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #666;">Advanced ensemble learning algorithm specifically tuned for market pattern recognition</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Always use random forest
        model_type = "random_forest"
    
    with col2:
        # Target period for predictions
        target_period = st.selectbox(
            "Training Target",
            options=["1W", "1M", "3M", "6M", "1Y"],
            index=1,  # Default to 1M
            key="target_period",
            help="The time period for which the model will be trained to predict returns"
        )
    
    # Add a forecast horizon selector
    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=30, 
        max_value=365,
        value=252,  # Default to 1 year
        step=30,
        help="Number of days to forecast into the future"
    )
    
    # Create a simple "Train Model" button
    train_model_button = st.button(
        "Train ML Model & Generate Forecast", 
        key="train_model_button",
        help="Train a machine learning model on historical market data to generate price forecasts"
    )
    
    # Handle model training
    if train_model_button:
        with st.spinner(f"Training {model_type} model for {target_period} forecasting..."):
            # Prepare features - focusing on major market events
            # Use the same drop threshold as set in the main application settings
            data, features = prepare_features(
                st.session_state.data,
                focus_on_drops=True,  # Focus on significant market events
                drop_threshold=-abs(st.session_state.drop_threshold)  # Use the same threshold from main settings
            )
            
            # Select target column for the model
            target_column = f'Fwd_Ret_{target_period}'
            
            if data.empty or len(features) == 0:
                st.error("Insufficient data for ML model training. Try adjusting the date range to include more market data.")
            elif target_column not in data.columns:
                st.error(f"Target column '{target_column}' not found in data. Try a different prediction target.")
            else:
                # Train the ML model
                model_result = train_model(
                    data, 
                    features, 
                    target_column, 
                    model_type=model_type, 
                    test_size=0.2
                )
                
                # Store the model in session state
                st.session_state.ml_models[target_period] = model_result
                
                if model_result['success']:
                    drop_events_count = len(data)
                    st.success(f"Model trained successfully on {drop_events_count} market events!")
                else:
                    st.error(f"Model training failed: {model_result.get('error', 'Unknown error')}")
    
    # Display the ML forecast
    st.markdown("#### S&P 500 Price Forecast")
    
    # Get the appropriate model from session state
    model_result = st.session_state.ml_models[target_period]
    
    if model_result is None:
        st.info("No forecast model available. Please train a model using the button above.")
        
        # Show a placeholder image or message
        st.markdown("""
        <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 8px; margin: 20px 0;">
            <h3 style="color: #6c757d;">ML Forecast Preview</h3>
            <p>Train a model to see a 1-year price projection with confidence intervals for 1W, 1M, 3M, and 1Y horizons.</p>
            <p style="font-style: italic; font-size: 0.9em; color: #6c757d;">
                The forecast will include YTD historical data and show confidence levels at key time intervals.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Get features for forecasting - using the same threshold from main settings
        _, features = prepare_features(
            st.session_state.data,
            focus_on_drops=True,
            drop_threshold=-abs(st.session_state.drop_threshold)
        )
        
        # Extract model target information
        model_target_column = model_result.get('target_column', '')
        target_period = model_target_column.replace('Fwd_Ret_', '') if 'Fwd_Ret_' in model_target_column else '1M'
        
        # Create and display the forecast chart
        forecast_chart = create_forecast_chart(
            model_result,
            st.session_state.data,
            features,
            days_to_forecast=forecast_days,
            title=f"S&P 500 {forecast_days}-Day Price Forecast",
            height=600
        )
        
        # Display the chart with enhanced styling
        st.markdown('<div class="forecast-chart">', unsafe_allow_html=True)
        st.plotly_chart(forecast_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add explanation of the forecast intervals
        st.markdown("""
        <div style="margin: 0.5rem 0 1.5rem 0; padding: 0.7rem; background-color: rgba(240, 248, 255, 0.6); border-radius: 5px; border-left: 3px solid #1E88E5;">
            <p style="margin: 0; font-size: 0.9rem; color: #1E4A7B;"><strong>About the Forecast:</strong> 
            This forecast shows the likely path of S&P 500 prices with confidence intervals widening over time. 
            The model identifies key timeframes (1W, 1M, 3M, 1Y) with specific price targets and confidence levels.
            Wider intervals indicate greater uncertainty in longer-term projections.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical indicators for current market conditions
    st.markdown("#### Key Technical Indicators")
    
    # Display technical indicators in a grid
    indicators = [
        {"name": "RSI (14)", "value": latest_data.get('RSI_14', 'N/A'), "color": "red" if latest_data.get('RSI_14', 50) > 70 else "green" if latest_data.get('RSI_14', 50) < 30 else "gray", "description": "Measures momentum - values >70 suggest overbought, <30 oversold"},
        {"name": "MACD", "value": latest_data.get('MACDh_12_26_9', 'N/A'), "color": "green" if latest_data.get('MACDh_12_26_9', 0) > 0 else "red", "description": "Trend indicator - positive values suggest bullish momentum"},
        {"name": "BB Position", "value": latest_data.get('BBP_20_2', 'N/A'), "color": "red" if latest_data.get('BBP_20_2', 0.5) > 1 else "green" if latest_data.get('BBP_20_2', 0.5) < 0 else "gray", "description": "Position within Bollinger Bands - >1 overbought, <0 oversold"},
        {"name": "ATR %", "value": latest_data.get('ATR_Pct', 'N/A'), "color": "orange" if latest_data.get('ATR_Pct', 1) > 3 else "blue", "description": "Measures volatility - higher values indicate more market volatility"}
    ]
    
    # Display indicators in a grid
    cols = st.columns(4)
    for i, indicator in enumerate(indicators):
        with cols[i]:
            value = indicator["value"]
            value_display = f"{value:.2f}" if isinstance(value, (int, float)) else value
            
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <p style="font-weight: 500; margin: 0; color: #1E4A7B;">{indicator['name']}</p>
                <p style="font-size: 22px; font-weight: 700; margin: 5px 0; color: {indicator['color']};">{value_display}</p>
                <p style="font-size: 11px; margin: 0; color: #6c757d;">{indicator['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Model performance metrics - only shown if a model is available
    if model_result is not None and model_result.get('success', False):
        st.markdown("#### Model Performance")
        
        metrics = model_result.get('metrics', {})
        
        # Display metrics in 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "R² Score (Test)", 
                f"{metrics.get('r2_test', 0):.3f}",
                delta=f"{metrics.get('r2_train', 0):.3f} (train)",
                delta_color="normal"  # Higher R² is better
            )
        
        with col2:
            st.metric(
                "Mean Absolute Error", 
                f"{metrics.get('mae_test', 0):.2f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "RMSE (Test Set)", 
                f"{metrics.get('rmse_test', 0):.2f}%",
                delta=None
            )
        
        # Feature importance chart - help user understand what drives the model
        feature_importance = create_feature_importance_chart(model_result, height=400)
        st.plotly_chart(feature_importance, use_container_width=True)