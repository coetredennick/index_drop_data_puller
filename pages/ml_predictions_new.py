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

def show_ml_predictions(market_data, ml_models_for_ticker, ml_model_params_for_ticker, active_ticker_symbol, active_ticker_name):
    """
    Display the Machine Learning Predictions tab focused on ML forecasting 
    with confidence intervals for different time periods (1W, 1M, 3M, 1Y)
    Shows a forecasting graph with YTD data and up to 1 year projection
    Parameters:
    -----------
    market_data : pd.DataFrame
        DataFrame containing historical market data for the active ticker.
    ml_models_for_ticker : dict
        Dictionary to store/retrieve trained ML models for the active ticker, keyed by forecast horizon.
    ml_model_params_for_ticker : dict
        Dictionary to store/retrieve ML model parameters for the active ticker, keyed by forecast horizon.
    active_ticker_symbol : str
        The symbol of the currently active ticker (e.g., "^GSPC").
    active_ticker_name : str
        The display name of the currently active ticker (e.g., "S&P 500").
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
    
    # Main title, now dynamic
    st.markdown(f"### {active_ticker_name} ML Forecasting")
    
    # Check if data is available (using the passed market_data)
    data_available = market_data is not None and not market_data.empty
    
    if not data_available:
        st.warning(f"No data available for {active_ticker_name}. Please adjust the date range and fetch data.")
        return
    
    # ml_models_for_ticker and ml_model_params_for_ticker are now passed in directly.
    # No need for: if 'ml_models' not in st.session_state: ...
    
    # Current market information from passed market_data
    latest_data = market_data.iloc[-1]
    latest_date = latest_data.name.strftime('%Y-%m-%d') if hasattr(latest_data.name, 'strftime') else str(latest_data.name)
    latest_close = latest_data['Close']
    latest_return = latest_data['Return'] if 'Return' in latest_data else 0
    
    # Display current market status prominently, now dynamic
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 8px; background-color: #f0f4f8; margin: 15px 0; border: 1px solid #d0e1f9;">
        <h3 style="margin: 0; color: #104E8B; font-size: 1.2rem;">Current {active_ticker_name}</h3>
        <div style="display: flex; align-items: center; margin-top: 8px;">
            <div style="font-size: 28px; font-weight: 700; margin-right: 15px;">${latest_close:.2f}</div>
            <div style="font-size: 16px; font-weight: 500; color: {'#21A366' if latest_return >= 0 else '#E63946'};">
                {latest_return:+.2f}% today
            </div>
        </div>
        <div style="font-size: 12px; color: #666; margin-top: 5px;">Last updated: {latest_date}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different model configurations / forecast horizons
    forecast_horizons = ["1W", "1M", "3M", "6M", "1Y"]
    # Make tab labels more descriptive if needed, e.g., f"{active_ticker_name} {horizon} Forecast"
    model_tabs = st.tabs([f"{horizon} Forecast" for horizon in forecast_horizons])

    for i, selected_horizon in enumerate(forecast_horizons):
        with model_tabs[i]:
            st.markdown(f"#### Configure and Train Model for {selected_horizon} Forecast - {active_ticker_name}")
            
            # Retrieve current parameters for this horizon, or defaults
            current_horizon_params = ml_model_params_for_ticker.get(selected_horizon, {})

            # --- Model Parameter Configuration UI in Sidebar (Example) ---
            # Note: Keys for UI elements must be unique across tabs and indices.
            # Using active_ticker_symbol and selected_horizon in keys ensures uniqueness.
            
            with st.sidebar.expander(f"Model Config: {active_ticker_name} - {selected_horizon}", expanded=(i==0)):
                st.markdown(f"**Parameters for {selected_horizon} ({active_ticker_name})**")
                lookback_period = st.slider(
                    label="Lookback Period (days):", 
                    min_value=21, max_value=756, 
                    value=current_horizon_params.get('lookback_period', 252),
                    step=21,
                    key=f"lookback_{active_ticker_symbol}_{selected_horizon}"
                )
                n_estimators = st.slider(
                    label="Number of Estimators (Trees):", 
                    min_value=50, max_value=500, 
                    value=current_horizon_params.get('n_estimators', 100),
                    step=50,
                    key=f"n_estimators_{active_ticker_symbol}_{selected_horizon}"
                )
                max_depth = st.slider(
                    label="Max Depth of Trees:", 
                    min_value=3, max_value=20, 
                    value=current_horizon_params.get('max_depth', 10),
                    step=1,
                    key=f"max_depth_{active_ticker_symbol}_{selected_horizon}"
                )
                learning_rate = st.select_slider(
                    label="Learning Rate:",
                    options=[0.01, 0.05, 0.1, 0.2],
                    value=current_horizon_params.get('learning_rate', 0.1),
                    key=f"learning_rate_{active_ticker_symbol}_{selected_horizon}"
                )
                # Add other parameters as needed, using dynamic keys and defaulting to current_horizon_params

                train_button_key = f"train_button_{active_ticker_symbol}_{selected_horizon}"
                if st.button(f"Train Model for {selected_horizon}", key=train_button_key):
                    with st.spinner(f"Training {active_ticker_name} {selected_horizon} model..."):
                        # Consolidate parameters for training
                        training_params = {
                            'lookback_period': lookback_period,
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            # Add other collected params
                        }
                        
                        # Store these parameters in the session state structure provided by app.py
                        ml_model_params_for_ticker[selected_horizon] = training_params
                        
                        # Prepare features and train the model using market_data
                        features_df, target_series = prepare_features(market_data, selected_horizon, lookback_period)
                        
                        if features_df is not None and not features_df.empty:
                            model_artifacts = train_model(features_df, target_series, training_params)
                            # Store the trained model artifacts (model, metrics, etc.)
                            ml_models_for_ticker[selected_horizon] = model_artifacts
                            st.success(f"{active_ticker_name} {selected_horizon} model trained successfully!")
                            st.rerun() # Rerun to update displays with new model
                        else:
                            st.error(f"Could not prepare features for {active_ticker_name} {selected_horizon} model.")
            
            # --- Display Model Results and Forecast --- 
            # Retrieve the potentially trained model for this horizon
            model_artifacts = ml_models_for_ticker.get(selected_horizon)

            if model_artifacts and model_artifacts.get('model'):
                st.markdown(f"##### {selected_horizon} Forecast for {active_ticker_name}")
                # Use model_artifacts (which includes model, metrics, predictions) to display charts
                # Example: create_forecast_chart, create_feature_importance_chart
                # Ensure these utility functions can handle the structure of model_artifacts
                
                # Example: Generating and displaying forecast chart
                # This part needs to be adapted based on what train_model returns and what create_forecast_chart expects
                # For instance, if predictions are part of model_artifacts:
                if 'predictions_df' in model_artifacts:
                    forecast_fig = create_forecast_chart(market_data, model_artifacts['predictions_df'], selected_horizon, active_ticker_name)
                    st.plotly_chart(forecast_fig, use_container_width=True)
                
                if 'metrics' in model_artifacts:
                    st.metric("Test R² Score", f"{model_artifacts['metrics'].get('r2_test', 0):.3f}")
                    # Display other metrics

                if 'feature_importances' in model_artifacts:
                    fi_chart = create_feature_importance_chart(model_artifacts, height=300)
                    st.plotly_chart(fi_chart, use_container_width=True)

            else:
                st.info(f"No model has been trained yet for the {selected_horizon} forecast for {active_ticker_name}. "
                        f"Please configure parameters in the sidebar and click 'Train Model'.")
            
            # Placeholder for the rest of the tab content which might include:
            # - Detailed prediction table
            # - Confidence intervals display
            # - Scenario analysis (if create_multi_scenario_forecast is used)
            # Ensure all st.session_state.data is replaced by market_data
            # Ensure all st.session_state.ml_models is replaced by ml_models_for_ticker
            # Ensure all UI element keys are dynamic (e.g., using active_ticker_symbol and selected_horizon)

    # The rest of the function, including sections like "Historical Drop Analysis based on ML Model" 
    # needs to be similarly refactored to use the passed parameters and dynamic keys.
    # This is a illustrative snippet of the required changes.

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
    selected_horizon = "1M" # default to 1M
    model_artifacts = ml_models_for_ticker.get(selected_horizon)

    if model_artifacts and model_artifacts.get('model'):
        st.markdown("#### Current Market Prediction")
        
        # Get features for prediction
        _, features = prepare_features(market_data, selected_horizon, lookback_period=252)
        
        # Get most recent data for prediction
        recent_data = market_data.tail(30)  # Use last 30 days of data for better feature calculation
        
        # Make prediction with enhanced function that returns a dictionary
        prediction_result = predict_returns(model_artifacts, recent_data, features)
        
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
            
            # Create a card style using Streamlit elements
            st.subheader(f"Predicted {selected_horizon} Return")
            
            # Create a container for the prediction card
            with st.container():
                # Layout with columns for prediction value and confidence
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.metric(
                        label="",
                        value=f"{prediction:.2f}%",
                        delta=None,
                    )
                    st.caption(f"Range: {lower_bound:.2f}% to {upper_bound:.2f}%")
                
                with col2:
                    st.text("Confidence: 95%")
                    # Use progress bar for confidence visualization
                    st.progress(0.95)
                    st.caption("Based on model accuracy")
                
                # Date and explanation
                date_str = prediction_date.strftime('%b %d, %Y') if isinstance(prediction_date, pd.Timestamp) else 'recent data'
                st.text(f"Based on market conditions analyzed on {date_str}")
                st.text(f"Forecast period: next {21} trading days")
                
                # Top contributing factors
                st.markdown("**Top Contributing Factors:**")
                
                # Display top contributing features in columns
                feature_cols = st.columns(len(top_features))
                for i, feature in enumerate(top_features):
                    with feature_cols[i]:
                        # Create a custom metric for each feature
                        st.markdown(f"**{feature['name']}**")
                        st.markdown(f"<span style='color: {feature['color']}; font-size: 16px; font-weight: bold;'>{feature['display']}</span>", unsafe_allow_html=True)
            
            # Add explanation of the confidence interval using a info box
            st.info(
                f"**Understanding the prediction:** The model predicts a {prediction:.2f}% return over the next {21} trading days, "
                f"with a 95% confidence interval of {lower_bound:.2f}% to {upper_bound:.2f}%. "
                f"This prediction considers current market conditions, technical indicators, VIX data, and trading volumes."
            )
            
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
                
                # Create a row of metrics using st.metric
                cols = st.columns(len(recovery_data))
                
                for i, (period, data) in enumerate(recovery_data.items()):
                    value = data["value"]
                    label = data["label"]
                    
                    if value is not None and not pd.isna(value):
                        if period != "Days":
                            # Format as percentage
                            formatted_value = f"{value:.2f}%"
                            delta_color = "normal" if value > 0 else "inverse"
                            delta = f"{value:.2f}%" if value != 0 else None
                            
                            # Display metric with Streamlit's component
                            with cols[i]:
                                st.metric(
                                    label=label,
                                    value=formatted_value,
                                    delta=delta,
                                    delta_color=delta_color,
                                    help="Post-drop recovery metric"
                                )
                        else:
                            # Days to recovery formatting
                            if value > 30:
                                formatted_value = ">30 days"
                            else:
                                formatted_value = f"{int(value)} days"
                            
                            # Display metric with Streamlit's component
                            with cols[i]:
                                st.metric(
                                    label=label,
                                    value=formatted_value,
                                    delta=None,
                                    help="Time to return to pre-drop levels"
                                )
                    else:
                        # Display placeholder for missing data
                        with cols[i]:
                            st.metric(
                                label=label,
                                value="N/A",
                                delta=None,
                                help="Data not available"
                            )
                
                # Add explanation for recovery metrics using a success info box
                st.success(
                    "**Understanding Recovery Metrics:** These metrics analyze market behavior following significant drops. "
                    '"Days To Recovery" shows the typical time it takes for the market to return to pre-drop levels, while return percentages '
                    "show how the market rebounds over specific time periods after drops."
                )
                
            except Exception as e:
                st.warning(f"Unable to display recovery metrics: {str(e)}")
            
        else:
            # Display simplified error message
            st.warning("Unable to make prediction with current market data. Try adjusting the settings or training a new model.")
            
        # Add prediction analysis chart
        prediction_chart = create_prediction_chart(model_artifacts, height=400)
        st.plotly_chart(prediction_chart, use_container_width=True)
    
    # Model performance metrics - only shown if a model is available
    if model_artifacts and model_artifacts.get('model'):
        st.markdown("#### Model Performance")
        
        metrics = model_artifacts.get('metrics', {})
        
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
        feature_importance = create_feature_importance_chart(model_artifacts, height=400)
        st.plotly_chart(feature_importance, use_container_width=True)