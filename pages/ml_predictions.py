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
    create_feature_importance_chart
)

def show_ml_predictions():
    """
    Display the Machine Learning Predictions tab with model training,
    performance analysis, and predictions for current market conditions
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
    if not st.session_state.data is not None or st.session_state.data.empty:
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
    
    # Model settings
    st.markdown("#### Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=["random_forest", "gradient_boosting", "linear_regression"],
            index=0,
            help="Select the type of machine learning model to train"
        )
    
    with col2:
        target_period = st.selectbox(
            "Prediction Target",
            options=["1W", "1M", "3M", "6M", "1Y"],
            index=1,
            help="Select the time horizon for return predictions"
        )
    
    with col3:
        test_size = st.slider(
            "Test Data Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing the model"
        )
    
    # Add market drop threshold controls
    col1, col2 = st.columns(2)
    
    with col1:
        drop_threshold = st.slider(
            "Market Drop Threshold (%)",
            min_value=-10.0,
            max_value=-1.0,
            value=-3.0,
            step=0.5,
            help="Minimum percentage drop to be considered a significant market event"
        )
    
    with col2:
        focus_on_drops = st.checkbox(
            "Focus on Market Drops",
            value=True,
            help="When enabled, the model will specifically focus on data from market drop events"
        )
    
    # Train model button
    if st.button("Train Model on Market Drops"):
        with st.spinner(f"Training {model_type} model for {target_period} returns after market drops..."):
            # Prepare features with drop focus
            data, features = prepare_features(
                st.session_state.data,
                focus_on_drops=focus_on_drops,
                drop_threshold=drop_threshold
            )
            
            # Select target column - using uppercase for consistency with data_fetcher.py
            target_column = f'Fwd_Ret_{target_period}'
            
            if target_column not in data.columns:
                st.error(f"Target column '{target_column}' not found in data.")
            else:
                # Train the model
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
                        'Drop (%)': event['magnitude'],
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
        
        return
    
    # Get the trained model
    model_result = st.session_state.ml_models[target_period]
    
    if not model_result['success']:
        st.error(f"Model error: {model_result.get('error', 'Unknown error')}")
        return
    
    # Model performance metrics
    st.markdown("#### Model Performance")
    
    metrics = model_result['metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mean Absolute Error (Test)", 
            f"{metrics['mae_test']:.2f}%",
            delta=f"{metrics['mae_train']:.2f}% (train)",
            delta_color="inverse"  # Lower MAE is better
        )
    
    with col2:
        st.metric(
            "Root Mean Squared Error (Test)", 
            f"{metrics['rmse_test']:.2f}%",
            delta=f"{metrics['rmse_train']:.2f}% (train)",
            delta_color="inverse"  # Lower RMSE is better
        )
    
    with col3:
        st.metric(
            "R² Score (Test)", 
            f"{metrics['r2_test']:.3f}",
            delta=f"{metrics['r2_train']:.3f} (train)",
            delta_color="normal"  # Higher R² is better
        )
    
    # Visualize model predictions vs actual returns
    st.markdown("#### Prediction Performance")
    
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
                
                # Display summary statistics
                st.markdown(f"""
                **Post-Drop Return Statistics ({target_period})**
                
                From {len(returns_series)} historical drop events:
                - **Positive Outcomes**: {positive_pct:.1f}% of events led to positive returns
                - **Average Return**: {avg_return:.2f}%
                - **Median Return**: {median_return:.2f}%
                - **Best Recovery**: +{best_return:.2f}%
                - **Worst Outcome**: {worst_return:.2f}%
                """)
                
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
                    
                    # Display analysis
                    st.markdown("""
                    **Drop Event Characteristics**
                    
                    The model was trained on various types of market drops:
                    """)
                    
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
    
    # Prepare features for current data - make sure to use same settings as the trained model
    current_data, features = prepare_features(
        st.session_state.data,
        focus_on_drops=False  # Don't filter current data, just prepare features
    )
    
    if current_data.empty:
        st.warning("Insufficient data to make predictions for current market conditions.")
        return
    
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
                <p style="margin:0; font-size:14px;">Based on current market conditions using {model_type.replace('_', ' ').title()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Confidence interval explanation
        st.markdown("""
        #### Prediction Interpretation
        
        The prediction shown is a point estimate based on current market conditions. Actual returns 
        can vary significantly due to unforeseen events and market conditions. Here's how to interpret 
        this prediction:
        
        - This prediction represents the expected return over the next {period}, not a guaranteed outcome.
        - The model is trained on historical data and past relationships between technical indicators and returns.
        - Market conditions outside the training data range may lead to less accurate predictions.
        - Always combine this prediction with other analysis tools and your own judgment.
        """.replace("{period}", target_period))
        
        # Prediction confidence based on model metrics
        r2 = metrics['r2_test']
        rmse = metrics['rmse_test']
        
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
    
    # Model details and limitations
    with st.expander("Model Details and Limitations"):
        st.markdown("""
        #### Model Details
        
        This machine learning model is trained to predict future returns based on current market 
        conditions and technical indicators. The model uses the following features:
        
        - Technical indicators (RSI, Stochastic, MACD, Bollinger Bands, etc.)
        - Price relative to moving averages
        - Recent price changes and volatility
        - Volume metrics
        
        #### Limitations
        
        Please be aware of the following limitations:
        
        1. **Past Performance**: The model is trained on historical data and assumes similar 
           relationships will hold in the future.
        2. **Black Swan Events**: The model cannot predict unexpected events like geopolitical crises, 
           pandemics, or other unpredictable market shocks.
        3. **Changing Market Regimes**: Market behavior can change over time, affecting the 
           reliability of predictions.
        4. **Limited Features**: The model only considers technical indicators and price patterns, 
           not fundamental data, news, or sentiment.
        
        #### Recommended Use
        
        This prediction tool should be used as one component of a broader investment decision-making 
        process, not as a standalone signal for trading decisions.
        """)
