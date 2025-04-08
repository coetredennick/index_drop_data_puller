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
    create_forecast_chart,
    create_multi_scenario_forecast
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
        <div style="padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 4px solid #4682B4; margin-bottom: 10px;">
            <p style="margin: 0; font-weight: 600; color: #104E8B;">🌲 Using Optimized Random Forest Model</p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #666;">Advanced ensemble learning algorithm specifically tuned for market pattern recognition</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VIX data integration info
        st.markdown("""
        <div style="padding: 10px; background-color: #fff8f0; border-radius: 5px; border-left: 4px solid #FF8C00;">
            <p style="margin: 0; font-weight: 600; color: #D35400;">📊 VIX Data Integration</p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #666;">Volatility Index data now enhances prediction accuracy by capturing market sentiment and fear</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Always use random forest
        model_type = "random_forest"
    
    with col2:
        # Target period for predictions
        target_period_options = ["1W", "1M", "3M", "6M", "1Y"]
        target_period_days = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
        
        target_period = st.selectbox(
            "Training Target",
            options=target_period_options,
            index=1,  # Default to 1M
            key="target_period",
            help="The time period for which the model will be trained to predict returns"
        )
    
    # Coordinated forecast horizon selector based on target period
    recommended_days = target_period_days.get(target_period, 252)
    
    # Ensure values are compatible with step
    min_val = max(5, recommended_days // 2)
    max_val = max(365, recommended_days * 4)
    step_val = max(5, recommended_days // 10)
    
    # Adjust recommended_days to be compatible with step
    recommended_days = (recommended_days // step_val) * step_val
    
    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=min_val, 
        max_value=max_val,
        value=recommended_days,  # Default to match training target
        step=step_val,
        help="Number of days to forecast into the future (auto-adjusted based on training target)"
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
    st.markdown("#### S&P 500 Multi-Scenario Forecast")
    
    # Get features for forecasting - using the same threshold from main settings
    with st.spinner("Generating multi-scenario forecast for all time periods..."):
        # Create simpler features with more error handling
        data = st.session_state.data.copy()
        
        # Make sure we have the bare minimum features
        if 'Return' not in data.columns:
            data['Return'] = data['Close'].pct_change() * 100
                
        if 'RSI_14' not in data.columns and len(data) > 14:
            # Calculate a simple RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI_14'] = 100 - (100 / (1 + rs))
                
        # Use a reduced feature set to avoid issues
        simplified_features = ['Return']
        
        if 'RSI_14' in data.columns:
            simplified_features.append('RSI_14')
            
        if 'Volume_Ratio_10D' in data.columns:
            simplified_features.append('Volume_Ratio_10D')
        elif 'Volume' in data.columns:
            # Create Volume_Ratio_10D if not available
            data['Volume_Ratio_10D'] = data['Volume'] / data['Volume'].rolling(10).mean()
            simplified_features.append('Volume_Ratio_10D')
        
        try:
            # Create multi-scenario forecast with simplified feature set
            with st.spinner("Generating multi-scenario market forecast..."):
                multi_scenario_chart = create_multi_scenario_forecast(
                    data,
                    simplified_features,
                    days_to_forecast=forecast_days,
                    title="S&P 500 Multiple Scenario Forecast (Bear, Base & Bull Cases)",
                    height=650
                )
                
                # Display the chart with enhanced styling
                st.markdown('<div class="forecast-chart">', unsafe_allow_html=True)
                st.plotly_chart(multi_scenario_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add explanation of the multiple scenario forecast
                st.markdown(f"""
                <div style="margin: 0.5rem 0 1.5rem 0; padding: 0.7rem; background-color: rgba(240, 248, 255, 0.6); border-radius: 5px; border-left: 3px solid #1E88E5;">
                    <p style="margin: 0; font-size: 0.9rem; color: #1E4A7B;"><strong>About the Multi-Scenario Forecast:</strong> 
                    This visualization shows three distinct market scenarios at key time intervals (1W, 1M, 3M, 6M, 1Y):</p>
                    <ul style="margin: 0.4rem 0 0.4rem 1.2rem; padding: 0; font-size: 0.9rem; color: #1E4A7B;">
                        <li><strong>Bear Case</strong> (20th percentile) - Based on performance of the worst 20% of historical periods</li>
                        <li><strong>Base Case</strong> (median) - Represents the most likely outcome based on median historical returns</li>
                        <li><strong>Bull Case</strong> (80th percentile) - Based on performance of the best 20% of historical periods</li>
                    </ul>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #666;">
                        <em>These scenarios are calculated using actual historical market returns rather than theoretical statistical ranges. 
                        They represent realistic market outcomes based on S&P 500 performance from 1950 to present.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            import traceback
            error_details = str(e)
            trace = traceback.format_exc()
            
            # Log the detailed error
            print(f"Error in multi-scenario forecast: {error_details}")
            print(trace)
            
            # Show user-friendly error message
            st.error(f"Could not generate multi-scenario forecast: {error_details[:100]}...")
            
            # Give the user more helpful information
            st.info("""
            **Troubleshooting suggestions:**
            1. Try adjusting the date range to include more recent data
            2. If the date range is very short, try extending it for better model training
            3. Use the model training button below to train a specific model
            """)
            
            # Create a simpler historical chart as fallback
            if data is not None and not data.empty and 'Close' in data.columns:
                st.subheader("Historical S&P 500 Performance (Fallback Chart)")
                recent_data = data.tail(90)  # Show last 90 days
                
                # Create simple price chart
                fig = px.line(recent_data, x=recent_data.index, y='Close', 
                             title="S&P 500 Recent Performance")
                fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)
            
    # Ensure we have a trained model - auto train if needed
    target_period = st.session_state.get("target_period", "1M")
    model_result = st.session_state.ml_models.get(target_period)
    
    # If no model is trained yet, auto-train one with default parameters
    if model_result is None or not model_result.get('success', False):
        try:
            with st.spinner(f"Auto-training {target_period} prediction model..."):
                st.session_state.drop_threshold = 3.0  # Default drop threshold
                st.session_state.model_type = "random_forest"  # Default model type
                st.session_state.focus_on_drops = True  # Default to focus on drops
                st.session_state.test_size = 0.2  # Default test size
                
                # Get features with current settings
                data_with_features, features = prepare_features(
                    st.session_state.data,
                    focus_on_drops=st.session_state.focus_on_drops,
                    drop_threshold=-abs(st.session_state.drop_threshold)
                )
                
                # Get the appropriate forward return column based on the target period
                target_periods = {"1W": "Fwd_Ret_1W", "1M": "Fwd_Ret_1M", "3M": "Fwd_Ret_3M", "1Y": "Fwd_Ret_1Y"}
                target_column = target_periods.get(target_period, "Fwd_Ret_1M")
                
                # Train the model
                model_result = train_model(
                    data_with_features,
                    features,
                    target_column,
                    model_type=st.session_state.model_type,
                    test_size=st.session_state.test_size
                )
                
                # Save the model to session state
                if model_result and model_result.get('success', False):
                    if 'ml_models' not in st.session_state:
                        st.session_state.ml_models = {}
                    st.session_state.ml_models[target_period] = model_result
                    print(f"Auto-trained {target_period} model successfully")
                else:
                    print(f"Failed to auto-train {target_period} model: {model_result.get('error', 'Unknown error')}")
        except Exception as e:
            import traceback
            print(f"Error auto-training model: {str(e)}")
            print(traceback.format_exc())
    
    # Technical indicators for current market conditions
    st.markdown("#### Key Technical Indicators")
    
    # Display technical indicators in a grid
    indicators = [
        {"name": "RSI (14)", "value": latest_data.get('RSI_14', 'N/A'), "color": "red" if latest_data.get('RSI_14', 50) > 70 else "green" if latest_data.get('RSI_14', 50) < 30 else "gray", "description": "Measures momentum - values >70 suggest overbought, <30 oversold"},
        {"name": "MACD", "value": latest_data.get('MACDh_12_26_9', 'N/A'), "color": "green" if latest_data.get('MACDh_12_26_9', 0) > 0 else "red", "description": "Trend indicator - positive values suggest bullish momentum"},
        {"name": "BB Position", "value": latest_data.get('BBP_20_2', 'N/A'), "color": "red" if latest_data.get('BBP_20_2', 0.5) > 1 else "green" if latest_data.get('BBP_20_2', 0.5) < 0 else "gray", "description": "Position within Bollinger Bands - >1 overbought, <0 oversold"},
        {"name": "ATR %", "value": latest_data.get('ATR_Pct', 'N/A'), "color": "orange" if latest_data.get('ATR_Pct', 1) > 3 else "blue", "description": "Measures volatility - higher values indicate more market volatility"},
        {"name": "VIX", "value": latest_data.get('VIX_Close', 'N/A'), "color": "red" if latest_data.get('VIX_Close', 20) > 30 else "orange" if latest_data.get('VIX_Close', 20) > 20 else "green", "description": "Volatility Index - higher values indicate higher expected volatility"}
    ]
    
    # Display indicators in a grid (5 columns now to include VIX)
    cols = st.columns(5)
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
    
    # Display ML prediction for current market
    if model_result is not None and model_result.get('success', False):
        st.markdown("#### Current Market Prediction")
        
        # Get features for prediction
        _, features = prepare_features(
            st.session_state.data,
            focus_on_drops=True, 
            drop_threshold=-abs(st.session_state.drop_threshold)
        )
        
        # Get most recent data for prediction
        recent_data = st.session_state.data.tail(30)  # Use last 30 days of data for better feature calculation
        
        # Make prediction with enhanced function that returns a dictionary
        prediction_result = predict_returns(model_result, recent_data, features)
        
        if prediction_result.get('success', False):
            # Extract key prediction information
            prediction = prediction_result.get('predicted_return')
            prediction_date = prediction_result.get('prediction_date')
            confidence_interval = prediction_result.get('confidence_interval_95', {})
            lower_bound = confidence_interval.get('lower')
            upper_bound = confidence_interval.get('upper')
            
            # Get feature contributions if available
            feature_contributions = prediction_result.get('feature_contributions', {})
            top_features = []
            
            if feature_contributions:
                # Sort features by absolute contribution and get top 3
                sorted_features = sorted(
                    feature_contributions.items(), 
                    key=lambda x: abs(x[1].get('scaled_contribution', 0)), 
                    reverse=True
                )[:3]
                
                for feat_name, feat_data in sorted_features:
                    contribution = feat_data.get('scaled_contribution', 0)
                    # Format feature name for display
                    display_name = feat_name.replace('_', ' ').title()
                    if "Vix" in display_name:
                        display_name = display_name.replace("Vix", "VIX")
                    if "Rsi" in display_name:
                        display_name = display_name.replace("Rsi", "RSI")
                    if "Macd" in display_name:
                        display_name = display_name.replace("Macd", "MACD")
                    
                    # Prepare contribution display with appropriate color
                    contrib_color = "green" if contribution > 0 else "red"
                    contrib_sign = "+" if contribution > 0 else ""
                    
                    top_features.append({
                        "name": display_name,
                        "contribution": contribution,
                        "color": contrib_color,
                        "display": f"{contrib_sign}{contribution:.2f}%"
                    })
            
            # Color coding based on prediction value
            color = "green" if prediction > 0 else "red"
            
            # Create a complete prediction card with top features included
            prediction_html = f"""
            <div style="padding: 20px; border-radius: 8px; background-color: white; box-shadow: 0 2px 6px rgba(0,0,0,0.15); margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h3 style="margin: 0 0 10px 0; font-size: 1.2rem; color: #1E4A7B;">Predicted {target_period} Return</h3>
                        <p style="font-size: 2.4rem; font-weight: 700; margin: 0; color: {color};">{prediction:.2f}%</p>
                        <p style="font-size: 1rem; margin: 5px 0 0 0; color: #6c757d;">
                            Range: <span style="color: #555; font-weight: 500;">{lower_bound:.2f}% to {upper_bound:.2f}%</span>
                        </p>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; min-width: 180px;">
                        <p style="font-size: 0.9rem; margin: 0 0 5px 0; color: #555; font-weight: 500;">Confidence: 95%</p>
                        <div style="height: 6px; background-color: #e9ecef; border-radius: 3px; margin-bottom: 10px;">
                            <div style="height: 100%; width: 95%; background-color: #4CAF50; border-radius: 3px;"></div>
                        </div>
                        <p style="font-size: 0.8rem; margin: 0; color: #6c757d;">Based on model accuracy</p>
                    </div>
                </div>
                
                <p style="font-size: 0.9rem; margin: 15px 0 15px 0; color: #666;">
                    Based on current market conditions analyzed on {prediction_date.strftime('%b %d, %Y') if isinstance(prediction_date, pd.Timestamp) else 'recent data'}, 
                    the model predicts this return for the S&P 500 over the next {target_period_days.get(target_period, 21)} trading days.
                </p>
                
                <div style="margin-top: 15px; background-color: #f9f9f9; padding: 12px; border-radius: 5px; border-left: 3px solid #1E88E5;">
                    <p style="font-size: 0.9rem; margin: 0 0 8px 0; color: #1E4A7B; font-weight: 500;">Top Contributing Factors:</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            """
            
            # Add top contributing features to the HTML
            for feature in top_features:
                prediction_html += f"""
                        <div style="background-color: white; border-radius: 4px; padding: 8px; flex: 1; min-width: 120px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                            <span style="font-size: 0.85rem; color: #555;">{feature['name']}</span>
                            <p style="font-size: 1rem; font-weight: 600; margin: 5px 0 0 0; color: {feature['color']};">{feature['display']}</p>
                        </div>
                """
            
            # Close all the container divs
            prediction_html += """
                    </div>
                </div>
            </div>
            """
            
            # Display the complete card
            st.markdown(prediction_html, unsafe_allow_html=True)
            
            # Add explanation of the confidence interval
            st.markdown(f"""
            <div style="margin: 10px 0 20px 0; padding: 10px; background-color: rgba(240, 248, 255, 0.5); border-radius: 5px; font-size: 0.85rem; color: #555;">
                <strong>Understanding the prediction:</strong> The model predicts a {prediction:.2f}% return over the next {target_period_days.get(target_period, 21)} trading days, 
                with a 95% confidence interval of {lower_bound:.2f}% to {upper_bound:.2f}%. 
                This prediction considers current market conditions, technical indicators, VIX data, and trading volumes.
            </div>
            """, unsafe_allow_html=True)
            
            # Display recovery metrics
            st.markdown("#### Recovery Analysis")
            
            # Get the latest data point for recovery metrics
            recovery_data = {}
            
            # Try to extract recovery metrics from recent data
            try:
                last_row = recent_data.iloc[-1]
                
                recovery_data = {
                    "1D": {"value": last_row.get("Recovery_1d", None), "label": "Next Day Return"},
                    "3D": {"value": last_row.get("Recovery_3d", None), "label": "3 Day Return"},
                    "5D": {"value": last_row.get("Recovery_5d", None), "label": "5 Day Return"},
                    "10D": {"value": last_row.get("Recovery_10d", None), "label": "10 Day Return"},
                    "Days": {"value": last_row.get("Days_To_Recovery", None), "label": "Days To Recovery"}
                }
                
                # Create a row of metrics
                cols = st.columns(len(recovery_data))
                
                for i, (period, data) in enumerate(recovery_data.items()):
                    value = data["value"]
                    label = data["label"]
                    
                    if value is not None and not pd.isna(value):
                        if period != "Days":
                            # Format as percentage with color
                            formatted_value = f"{value:.2f}%"
                            color = "green" if value > 0 else "red"
                            delta_value = None
                        else:
                            # Days to recovery formatting
                            if value > 30:
                                formatted_value = ">30 days"
                                color = "orange"
                            else:
                                formatted_value = f"{value:.0f} days"
                                color = "blue"
                            delta_value = None
                        
                        # Display the metric
                        with cols[i]:
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 5px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <p style="font-weight: 500; margin: 0; color: #1E4A7B;">{label}</p>
                                <p style="font-size: 22px; font-weight: 700; margin: 5px 0; color: {color};">{formatted_value}</p>
                                <p style="font-size: 11px; margin: 0; color: #6c757d;">Post-drop recovery metric</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Display placeholder for missing data
                        with cols[i]:
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 5px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <p style="font-weight: 500; margin: 0; color: #1E4A7B;">{label}</p>
                                <p style="font-size: 22px; font-weight: 700; margin: 5px 0; color: #6c757d;">N/A</p>
                                <p style="font-size: 11px; margin: 0; color: #6c757d;">Data not available</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add explanation for recovery metrics
                st.markdown("""
                <div style="margin: 10px 0 20px 0; padding: 10px; background-color: rgba(240, 255, 240, 0.5); border-radius: 5px; font-size: 0.85rem; color: #555;">
                    <strong>Understanding Recovery Metrics:</strong> These metrics analyze market behavior following significant drops. 
                    "Days To Recovery" shows the typical time it takes for the market to return to pre-drop levels, while return percentages 
                    show how the market rebounds over specific time periods after drops.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Unable to display recovery metrics: {str(e)}")
            
        else:
            # Display simplified error message
            st.warning("Unable to make prediction with current market data. Try adjusting the settings or training a new model.")
            
        # Add prediction analysis chart
        prediction_chart = create_prediction_chart(model_result, height=400)
        st.plotly_chart(prediction_chart, use_container_width=True)
    
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