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

from utils.ml_models import (
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
    
    # Add custom styling for the ML Predictions page
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
        
        /* Style for selectbox */
        div[data-testid="stSelectbox"] label {
            font-weight: 500;
            color: #1E4A7B;
        }
        
        /* Metric styling */
        div[data-testid="stMetric"] {
            background-color: white;
            padding: 0.7rem;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Model metrics card */
        .metrics-card {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Prediction card */
        .prediction-card {
            text-align: center;
            background-color: white;
            border-radius: 5px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Prediction value */
        .prediction-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        /* Prediction label */
        .prediction-label {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
        }
        
        /* Feature table */
        .feature-table {
            font-size: 0.9rem;
        }
        
        /* Model info */
        .model-info {
            font-size: 0.8rem;
            color: #6c757d;
            font-style: italic;
            text-align: center;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("### Machine Learning Predictions")
    
    # Check if data is available
    data_available = st.session_state.data is not None and not st.session_state.data.empty
    
    if not data_available:
        st.warning("No data available. Please adjust the date range and fetch data.")
    # Continue only if data is available (wrapped in conditional blocks instead of early return)
    
    # Initialize session state for ML models
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = {
            '1W': None,
            '1M': None,
            '3M': None,
            '6M': None,
            '1Y': None
        }
    
    # Model settings
    st.markdown("#### Model Configuration")
    
    # Initialize or get form state from session state
    if 'ml_model_type' not in st.session_state:
        st.session_state.ml_model_type = "random_forest"
    if 'ml_target_period' not in st.session_state:
        st.session_state.ml_target_period = "1M"
    if 'ml_test_size' not in st.session_state:
        st.session_state.ml_test_size = 0.2
    if 'ml_drop_threshold' not in st.session_state:
        st.session_state.ml_drop_threshold = -3.0
    if 'ml_focus_on_drops' not in st.session_state:
        st.session_state.ml_focus_on_drops = True
    
    # Use Streamlit callbacks instead of forms to avoid page reloads
    # Define callback functions for each control
    def update_model_type():
        st.session_state.ml_model_type = st.session_state.cb_model_type
    
    def update_target_period():
        st.session_state.ml_target_period = st.session_state.cb_target_period
    
    def update_test_size():
        st.session_state.ml_test_size = st.session_state.cb_test_size
    
    def update_drop_threshold():
        st.session_state.ml_drop_threshold = st.session_state.cb_drop_threshold
    
    def update_focus_on_drops():
        st.session_state.ml_focus_on_drops = st.session_state.cb_focus_on_drops
    
    # Create controls with callbacks
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=["random_forest", "gradient_boosting", "linear_regression"],
            index=["random_forest", "gradient_boosting", "linear_regression"].index(st.session_state.ml_model_type),
            key="cb_model_type",
            on_change=update_model_type
        )
    
    with col2:
        target_period = st.selectbox(
            "Prediction Target",
            options=["1W", "1M", "3M", "6M", "1Y"],
            index=["1W", "1M", "3M", "6M", "1Y"].index(st.session_state.ml_target_period),
            key="cb_target_period",
            on_change=update_target_period
        )
    
    with col3:
        test_size = st.slider(
            "Test Data Size",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.ml_test_size,
            step=0.05,
            key="cb_test_size",
            on_change=update_test_size
        )
    
    # Add market drop threshold controls
    col1, col2 = st.columns(2)
    
    with col1:
        drop_threshold = st.slider(
            "Market Drop Threshold (%)",
            min_value=-10.0,
            max_value=-1.0,
            value=st.session_state.ml_drop_threshold,
            step=0.5,
            key="cb_drop_threshold",
            on_change=update_drop_threshold
        )
    
    with col2:
        focus_on_drops = st.checkbox(
            "Focus on Market Drops",
            value=st.session_state.ml_focus_on_drops,
            key="cb_focus_on_drops",
            on_change=update_focus_on_drops
        )
    
    # Create a button outside of a form
    def train_model_callback():
        st.session_state.train_model_clicked = True
    
    train_model_button = st.button(
        "Train Model on Market Drops", 
        on_click=train_model_callback,
        key="train_model_button"
    )
    
    # Process button click using session state
    if 'train_model_clicked' not in st.session_state:
        st.session_state.train_model_clicked = False
        
    # Model training is triggered by this flag
    if st.session_state.train_model_clicked:
        # Reset the flag for next time
        st.session_state.train_model_clicked = False
        with st.spinner(f"Training {model_type} model for {target_period} returns after market drops..."):
            # Prepare features with drop focus
            data, features = prepare_features(
                st.session_state.data,
                focus_on_drops=focus_on_drops,
                drop_threshold=drop_threshold
            )
            
            # Select target column - using uppercase for consistency with data_fetcher.py
            target_column = f'Fwd_Ret_{target_period}'
            
            if data.empty or len(features) == 0:
                st.error("Insufficient data for ML model training. Try adjusting the drop threshold to include more market events.")
            elif target_column not in data.columns:
                available_columns = [col for col in data.columns if col.startswith('Fwd_Ret_')]
                if available_columns:
                    # Use a different target column if available
                    alternative_column = available_columns[0]
                    st.warning(f"Target column '{target_column}' not found in filtered data. Using '{alternative_column}' instead.")
                    target_column = alternative_column
                    
                    # Train the model with alternative column
                    model_result = train_model(
                        data,
                        features,
                        target_column,
                        model_type=model_type,
                        test_size=test_size
                    )
                    
                    # Store the model in session state with modified target period
                    # Extract the period from the column name (e.g., 'Fwd_Ret_1M' → '1M')
                    actual_period = target_column.replace('Fwd_Ret_', '')
                    st.session_state.ml_models[actual_period] = model_result
                    st.session_state.drop_training_data = data
                    
                    if model_result['success']:
                        # Count number of drop events
                        drop_events_count = len(data)
                        st.success(f"Model trained successfully on {drop_events_count} market drop events for {actual_period} returns instead!")
                    else:
                        st.error(f"Model training failed: {model_result.get('error', 'Unknown error')}")
                else:
                    st.error(f"No forward return data available. Try adjusting the date range or drop threshold.")
            else:
                # Train the model with the requested column
                model_result = train_model(
                    data, 
                    features, 
                    target_column, 
                    model_type=model_type, 
                    test_size=test_size
                )
                
                # Store the model in session state
                st.session_state.ml_models[target_period] = model_result
                st.session_state.drop_training_data = data
                
                if model_result['success']:
                    # Count number of drop events
                    drop_events_count = len(data)
                    st.success(f"Model trained successfully on {drop_events_count} market drop events for {target_period} returns!")
                else:
                    st.error(f"Model training failed: {model_result.get('error', 'Unknown error')}")
    
    # Current market conditions assessment
    latest_data = st.session_state.data.iloc[-1]
    latest_date = latest_data.name.strftime('%Y-%m-%d') if hasattr(latest_data.name, 'strftime') else str(latest_data.name)
    latest_close = latest_data['Close']
    latest_return = latest_data['Return'] if 'Return' in latest_data else 0
    
    # Check if we are in a drop event based on threshold
    is_current_drop = latest_return <= drop_threshold
    
    # Display current market status
    st.markdown("#### Current Market Status")
    cols = st.columns([2, 1, 1])
    
    with cols[0]:
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; background-color: {'#f8d7da' if is_current_drop else '#d4edda'}; margin-bottom: 15px;">
            <h3 style="margin:0; color: {'#721c24' if is_current_drop else '#155724'};">
                {'⚠️ Market Drop Detected' if is_current_drop else '✅ Normal Market Conditions'}
            </h3>
            <p style="margin-top:5px; margin-bottom:0;">
                S&P 500: ${latest_close:.2f} | Daily change: {latest_return:.2f}% | Date: {latest_date}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent market activity
    recent_data = st.session_state.data.tail(10)
    recent_returns = recent_data['Return'].tolist() if 'Return' in recent_data else [0]
    min_recent_return = min(recent_returns)
    
    with cols[1]:
        st.metric("10-Day Min Return", f"{min_recent_return:.2f}%", delta=None)
    
    with cols[2]:
        volatility = recent_data['Return'].std() if 'Return' in recent_data else 0
        st.metric("10-Day Volatility", f"{volatility:.2f}%", delta=None)
    
    # Check if a model has been trained
    if st.session_state.ml_models[target_period] is None:
        # Show historical drop statistics instead
        st.markdown("#### Historical Market Drop Analysis")
        st.info("""
        No model trained yet. Please train a model using the button above.
        
        While we don't have a model yet, here's what historical data tells us about market drops:
        """)
        
        # Calculate some basic statistics about drops
        from utils.event_detection import detect_drop_events
        
        if 'Return' in st.session_state.data.columns:
            # Detect drop events using the threshold
            drop_events = detect_drop_events(st.session_state.data, abs(drop_threshold))
            
            if drop_events:
                num_events = len(drop_events)
                
                # Create a dataframe with returns after drops
                returns_after_drops = []
                for event in drop_events:
                    event_date = event['date']
                    event_idx = st.session_state.data.index.get_loc(event_date)
                    
                    # Calculate forward returns if possible
                    fwd_returns = {}
                    for days, period in [(5, '1W'), (21, '1M'), (63, '3M')]:
                        if event_idx + days < len(st.session_state.data):
                            start_price = st.session_state.data['Close'].iloc[event_idx]
                            end_price = st.session_state.data['Close'].iloc[event_idx + days]
                            fwd_return = (end_price / start_price - 1) * 100
                            fwd_returns[period] = fwd_return
                    
                    returns_after_drops.append({
                        'Date': event_date,
                        'Drop (%)': event['drop_pct'],
                        'Return 1W (%)': fwd_returns.get('1W', None),
                        'Return 1M (%)': fwd_returns.get('1M', None),
                        'Return 3M (%)': fwd_returns.get('3M', None)
                    })
                
                # Convert to dataframe
                if returns_after_drops:
                    df_returns = pd.DataFrame(returns_after_drops)
                    
                    # Calculate statistics
                    positive_1w = (df_returns['Return 1W (%)'] > 0).mean() * 100
                    positive_1m = (df_returns['Return 1M (%)'] > 0).mean() * 100
                    positive_3m = (df_returns['Return 3M (%)'] > 0).mean() * 100
                    
                    avg_1w = df_returns['Return 1W (%)'].mean()
                    avg_1m = df_returns['Return 1M (%)'].mean()
                    avg_3m = df_returns['Return 3M (%)'].mean()
                    
                    # Display insights
                    st.markdown(f"""
                    Based on analysis of {num_events} historical drop events of {abs(drop_threshold)}% or more:
                    
                    - **1 Week Later**: {positive_1w:.1f}% of drops led to positive returns (Avg: {avg_1w:.2f}%)
                    - **1 Month Later**: {positive_1m:.1f}% of drops led to positive returns (Avg: {avg_1m:.2f}%)
                    - **3 Months Later**: {positive_3m:.1f}% of drops led to positive returns (Avg: {avg_3m:.2f}%)
                    """)
                    
                    # Show distribution chart
                    import plotly.express as px
                    
                    # Create a distribution chart for the selected period
                    selected_period = 'Return 1M (%)' # Default to 1M
                    if target_period == '1W':
                        selected_period = 'Return 1W (%)'
                    elif target_period == '3M':
                        selected_period = 'Return 3M (%)'
                    
                    fig = px.histogram(
                        df_returns, 
                        x=selected_period,
                        nbins=30,
                        title=f"Distribution of {target_period} Returns After {abs(drop_threshold)}%+ Drops",
                        labels={selected_period: f'{target_period} Return (%)'},
                        color_discrete_sequence=['rgba(0, 0, 255, 0.6)']
                    )
                    
                    # Add vertical line at 0
                    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the most recent drop events
                    st.markdown("#### Recent Market Drop Events")
                    st.dataframe(
                        df_returns.sort_values('Date', ascending=False).head(10),
                        use_container_width=True
                    )
            else:
                st.warning(f"No market drops of {abs(drop_threshold)}% or more detected in the selected date range.")
        
        # Continue to show placeholder content for untrained model instead of returning
    
    # Get the trained model
    model_result = st.session_state.ml_models[target_period]
    
    # Check if model is None or training failed
    if model_result is None:
        # No model trained yet, show placeholder
        st.info("No model has been trained yet. Use the 'Train Model on Market Drops' button above to train a model.")
        # Empty placeholder
        # Skip the rest of the model-dependent sections instead of returning
        st.markdown("### S&P 500 Price Forecast")
        st.info("No forecast available until a model is trained.")
        st.markdown("#### Current Market Prediction")
        st.info("No predictions available until a model is trained.")
    
    elif not model_result['success']:
        st.error(f"Model error: {model_result.get('error', 'Unknown error')}")
        # Display alternative content instead of return
        st.markdown("#### Model Training Error")
        # Create placeholder structure for the rest of the page
        st.markdown("### S&P 500 Price Forecast")
        st.info("No forecast available due to model training error.")
        st.markdown("#### Model Performance")
        st.info("No performance metrics available.")
        st.markdown("#### Current Market Prediction")
        st.info("No predictions available.")
        # Skip the rest of the prediction logic - don't return
    
    # Only show these sections if model was successful
    if model_result is not None and model_result.get('success', False):
        # Add S&P 500 Price Forecast
        st.markdown("### S&P 500 Price Forecast")
        
        # Create forecast chart
        forecast_days_options = {
            "7 Days": 7,
            "14 Days": 14,
            "30 Days": 30,
            "60 Days": 60,
            "90 Days": 90
        }
        
        forecast_col1, forecast_col2 = st.columns([4, 1])
        
        with forecast_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some space
            forecast_days = st.selectbox(
                "Forecast Period",
                list(forecast_days_options.keys()),
                index=2  # Default to 30 days
            )
            days_to_forecast = forecast_days_options[forecast_days]
        
        with forecast_col1:
            # Prepare features for forecasting
            current_data, features = prepare_features(
                st.session_state.data,
                focus_on_drops=False  # Don't filter for forecasting
            )
            
            # Add specific insights about what the model is using for forecasting
            model_target_column = model_result.get('target_column', 'Fwd_Ret_1M')
            target_period = model_target_column.replace('Fwd_Ret_', '') if 'Fwd_Ret_' in model_target_column else '1M'
            st.info(f"Using ML model trained on {target_period} returns for S&P 500 price forecasting")
            
            # Get the most recent price for reference
            latest_price = st.session_state.data["Close"].iloc[-1] if not st.session_state.data.empty else 0
            latest_date = st.session_state.data.index[-1] if not st.session_state.data.empty else "N/A"
            
            # Show current price information
            price_col1, price_col2 = st.columns([1, 2])
            with price_col1:
                st.metric(
                    label="Latest S&P 500 Price", 
                    value=f"${latest_price:.2f}",
                    delta=f"{st.session_state.data['Return'].iloc[-1]:.2f}%" if not st.session_state.data.empty else None
                )
            with price_col2:
                st.caption(f"As of {latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else latest_date}")
                st.caption("The ML forecast is based on this starting price point")
            
            # Create and display the updated ML forecast chart
            forecast_chart = create_forecast_chart(
                model_result,
                st.session_state.data,
                features,
                days_to_forecast=days_to_forecast,
                title=f"S&P 500 {forecast_days}-Day ML Price Forecast"
            )
            st.plotly_chart(forecast_chart, use_container_width=True)
        
        # No explanatory text
        
        st.markdown("---")
        
        # Model performance metrics
        st.markdown("#### Model Performance")
        
        metrics = model_result.get('metrics', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mean Absolute Error (Test)", 
                f"{metrics.get('mae_test', 0):.2f}%",
                delta=f"{metrics.get('mae_train', 0):.2f}% (train)",
                delta_color="inverse"  # Lower MAE is better
            )
        
        with col2:
            st.metric(
                "Root Mean Squared Error (Test)", 
                f"{metrics.get('rmse_test', 0):.2f}%",
                delta=f"{metrics.get('rmse_train', 0):.2f}% (train)",
                delta_color="inverse"  # Lower RMSE is better
            )
        
        with col3:
            st.metric(
                "R² Score (Test)", 
                f"{metrics.get('r2_test', 0):.3f}",
                delta=f"{metrics.get('r2_train', 0):.3f} (train)",
                delta_color="normal"  # Higher R² is better
            )
    
    # Visualize model predictions vs actual returns
    if model_result is not None and model_result.get('success', False):
        st.markdown("#### Prediction Performance")
        
        # Get target period from model result
        model_target_column = model_result.get('target_column', 'Fwd_Ret_1M')
        target_period = model_target_column.replace('Fwd_Ret_', '') if 'Fwd_Ret_' in model_target_column else '1M'
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction vs actual scatter plot
            pred_chart = create_prediction_chart(
                model_result, 
                title=f"Model Predictions vs Actual {target_period} Returns"
            )
            st.plotly_chart(pred_chart, use_container_width=True)
        
        with col2:
            # Feature importance chart
            feat_chart = create_feature_importance_chart(
                model_result, 
                title="Feature Importance"
            )
            st.plotly_chart(feat_chart, use_container_width=True)
    
    # Add drop events analysis from model training data
    if 'drop_training_data' in st.session_state and focus_on_drops:
        st.markdown("#### Market Drop Analysis")
        
        drop_data = st.session_state.drop_training_data
        
        # Create tab view
        tab1, tab2 = st.tabs(["Post-Drop Returns", "Event Characteristics"])
        
        with tab1:
            # Calculate statistics for the selected period
            target_col = f'Fwd_Ret_{target_period}'
            if target_col in drop_data.columns:
                returns_series = drop_data[target_col].dropna()
                
                # Basic statistics
                positive_pct = (returns_series > 0).mean() * 100
                avg_return = returns_series.mean()
                median_return = returns_series.median()
                best_return = returns_series.max()
                worst_return = returns_series.min()
                
                # Display compact statistics without explanation
                st.markdown(f"**Post-Drop Return Statistics ({target_period})**")
                
                # Create a more compact metrics display
                metric_cols = st.columns(5)
                with metric_cols[0]:
                    st.metric("Positive %", f"{positive_pct:.1f}%")
                with metric_cols[1]:
                    st.metric("Avg Return", f"{avg_return:.2f}%")
                with metric_cols[2]:
                    st.metric("Median", f"{median_return:.2f}%")
                with metric_cols[3]:
                    st.metric("Best", f"+{best_return:.2f}%")
                with metric_cols[4]:
                    st.metric("Worst", f"{worst_return:.2f}%")
                
                # Create a histogram
                import plotly.express as px
                fig = px.histogram(
                    returns_series,
                    nbins=30,
                    title=f"Distribution of {target_period} Returns After Market Drops",
                    labels={'value': f'{target_period} Return (%)'},
                    color_discrete_sequence=['rgba(0, 0, 255, 0.6)']
                )
                
                # Add vertical line at 0
                fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Examine drop characteristics
            if 'Drop_Streak' in drop_data.columns:
                # Calculate distributions
                streak_counts = drop_data['Drop_Streak'].value_counts().sort_index()
                
                if 'Cumulative_Drop' in drop_data.columns:
                    # Group by drop streak
                    drop_magnitudes = drop_data.groupby('Drop_Streak')['Cumulative_Drop'].mean()
                    
                    # Display title only
                    st.markdown("**Drop Event Characteristics**")
                    
                    # Create combined dataframe
                    if not streak_counts.empty:
                        streak_stats = pd.DataFrame({
                            'Consecutive Drop Days': streak_counts.index,
                            'Number of Events': streak_counts.values
                        })
                        
                        # Add average magnitude
                        streak_stats['Avg. Magnitude (%)'] = [
                            drop_magnitudes.get(days, 0) for days in streak_stats['Consecutive Drop Days']
                        ]
                        
                        # Display as table
                        st.dataframe(streak_stats)
                        
                        # Create feature relationship charts
                        if target_col in drop_data.columns:
                            st.subheader("Feature Relationships with Returns")
                            
                            # Create relationship plots
                            plot_cols = st.columns(2)
                            
                            with plot_cols[0]:
                                # Relationship between drop streak and returns
                                import plotly.express as px
                                fig = px.box(
                                    drop_data,
                                    x='Drop_Streak',
                                    y=target_col,
                                    title=f"Returns by Consecutive Drop Days",
                                    labels={
                                        'Drop_Streak': 'Consecutive Drop Days',
                                        target_col: f'{target_period} Return (%)'
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with plot_cols[1]:
                                # Relationship between RSI and returns
                                if 'RSI_14' in drop_data.columns:
                                    import plotly.express as px
                                    fig = px.scatter(
                                        drop_data,
                                        x='RSI_14',
                                        y=target_col,
                                        title=f"Returns by RSI Level",
                                        labels={
                                            'RSI_14': 'RSI (14)',
                                            target_col: f'{target_period} Return (%)'
                                        },
                                        trendline='ols'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
    
    # Current market prediction
    st.markdown("#### Current Market Prediction")
    
    # Check if model_result exists and is valid before proceeding
    if model_result is not None and model_result.get('success', False):
        # Prepare features for current data - make sure to use same settings as the trained model
        current_data, features = prepare_features(
            st.session_state.data,
            focus_on_drops=False  # Don't filter current data, just prepare features
        )
        
        if current_data.empty:
            st.warning("Insufficient data to make predictions for current market conditions.")
            # Simplified message
            st.markdown("#### No prediction available")
            # Continue to rest of the page - don't return
        else:
            # Continue with prediction logic for valid data
            # This section will only execute if we have a valid model and data
            metrics = model_result.get('metrics', {})
            
            # Make prediction for the latest data point
            prediction = predict_returns(model_result, current_data, features)
            
            if prediction is not None:
                # Determine color based on prediction
                if prediction > 0:
                    prediction_color = "green"
                else:
                    prediction_color = "red"
                
                # Create a styled card for the prediction
                st.markdown(
                    f"""
                    <div style="border:1px solid {prediction_color}; border-radius:5px; padding:20px; background-color:rgba({0 if prediction_color == 'green' else 255}, {128 if prediction_color == 'green' else 0}, 0, 0.1); text-align:center; margin-bottom:20px;">
                        <h3 style="margin:0; color:{prediction_color};">Predicted {target_period} Return</h3>
                        <div style="font-size:48px; font-weight:bold; margin:10px 0; color:{prediction_color};">{prediction:.2f}%</div>
                        <p style="margin:0; font-size:14px;">Based on current market conditions using {model_result.get('model_type', 'Random Forest').replace('_', ' ').title()}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # No interpretation explanation to keep UI clean
                
                # Prediction confidence based on model metrics
                r2 = metrics.get('r2_test', 0)
                rmse = metrics.get('rmse_test', 0)
                
                # Simple confidence assessment based on R²
                if r2 > 0.3:
                    confidence = "High"
                    confidence_color = "green"
                elif r2 > 0.1:
                    confidence = "Medium"
                    confidence_color = "orange"
                else:
                    confidence = "Low"
                    confidence_color = "red"
                
                st.markdown(
                    f"""
                    <div style="border:1px solid {confidence_color}; border-radius:5px; padding:10px; background-color:rgba({0 if confidence_color == 'green' else (255 if confidence_color == 'red' else 255)}, {128 if confidence_color == 'green' else (0 if confidence_color == 'red' else 165)}, 0, 0.1);">
                        <h4 style="margin:0; color:{confidence_color};">Prediction Confidence: {confidence}</h4>
                        <p style="margin:5px 0; font-size:12px;">
                            Based on model R² score of {r2:.3f} and RMSE of {rmse:.2f}%.<br>
                            The actual return is likely to be within ±{(rmse * 1.96):.2f}% of the prediction (95% confidence interval).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("Unable to make a prediction for current market conditions.")
    
    # No model details expander to keep interface clean