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
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner(f"Training {model_type} model for {target_period} returns..."):
            # Prepare features
            data, features = prepare_features(st.session_state.data)
            
            # Select target column
            target_column = f'Fwd_Ret_{target_period.lower()}'
            
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
                
                if model_result['success']:
                    st.success(f"Model trained successfully for {target_period} returns!")
                else:
                    st.error(f"Model training failed: {model_result.get('error', 'Unknown error')}")
    
    # Check if a model has been trained
    if st.session_state.ml_models[target_period] is None:
        st.warning("No model trained yet. Please train a model using the button above.")
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
    
    # Current market prediction
    st.markdown("#### Current Market Prediction")
    
    # Prepare features for current data
    current_data, features = prepare_features(st.session_state.data)
    
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
