import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_features(data):
    """
    Prepare features for machine learning models
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with technical indicators
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features for ML models
    """
    # Create a copy of the data
    df = data.copy()
    
    # Select features for the model
    features = [
        'RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Volume_Ratio'
    ]
    
    # Add price-based features
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']
        features.append('SMA_50_200_Ratio')
    
    if 'EMA_20' in df.columns:
        df['EMA_20_Close_Ratio'] = df['EMA_20'] / df['Close']
        features.append('EMA_20_Close_Ratio')
    
    # Add lagged returns
    for lag in [1, 5, 10, 20]:
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
        features.append(f'Return_Lag_{lag}')
    
    # Calculate rolling volatility
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    features.append('Volatility_20')
    
    # Remove rows with NaN values
    df = df.dropna(subset=features)
    
    return df, features

def train_model(data, features, target_column, model_type='random_forest', test_size=0.2):
    """
    Train a machine learning model to predict future returns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing features and target variable
    features : list
        List of feature column names
    target_column : str
        Name of the target column (e.g., 'Fwd_Ret_1M')
    model_type : str, optional
        Type of model to train ('random_forest', 'gradient_boosting', or 'linear_regression')
    test_size : float, optional
        Proportion of data to use for testing
        
    Returns:
    --------
    dict
        Dictionary containing the trained model and performance metrics
    """
    # Remove rows with NaN in the target column
    df = data.dropna(subset=[target_column])
    
    if df.empty:
        return {
            'success': False,
            'error': 'No data available after removing NaN values'
        }
    
    # Split features and target
    X = df[features]
    y = df[target_column]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Initialize the model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    else:
        return {
            'success': False,
            'error': f'Unknown model type: {model_type}'
        }
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test)
    }
    
    # Calculate feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.coef_
        }).sort_values(by='Importance', ascending=False)
    
    return {
        'success': True,
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

def predict_returns(model_result, current_data, features):
    """
    Make predictions using the trained model for current market conditions
    
    Parameters:
    -----------
    model_result : dict
        Dictionary containing the trained model and performance metrics
    current_data : pandas.DataFrame
        DataFrame containing current market data
    features : list
        List of feature column names
        
    Returns:
    --------
    float
        Predicted return
    """
    if not model_result['success']:
        return None
    
    # Extract the model
    model = model_result['model']
    
    # Prepare current data
    current_features = current_data[features].iloc[-1].values.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(current_features)[0]
    
    return prediction

def create_prediction_chart(model_result, title="Model Predictions vs Actual Returns", height=400):
    """
    Create a chart comparing model predictions with actual returns
    
    Parameters:
    -----------
    model_result : dict
        Dictionary containing the trained model and performance metrics
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Prediction chart
    """
    if not model_result['success']:
        # No model available
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No model available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Extract test data and predictions
    y_test = model_result['y_test']
    y_pred = model_result['y_pred_test']
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot of actual vs predicted values
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(0, 0, 255, 0.6)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name="Test Samples",
        )
    )
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name="Perfect Prediction"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Actual Returns (%)",
        yaxis_title="Predicted Returns (%)",
        height=height,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    return fig

def create_feature_importance_chart(model_result, title="Feature Importance", height=400):
    """
    Create a chart showing feature importance from the model
    
    Parameters:
    -----------
    model_result : dict
        Dictionary containing the trained model and performance metrics
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Feature importance chart
    """
    if not model_result['success']:
        # No model available
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No model available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Extract feature importance
    feature_importance = model_result['feature_importance']
    
    # Sort by importance for better visualization
    feature_importance = feature_importance.sort_values(by='Importance', ascending=True)
    
    # Create figure
    fig = go.Figure()
    
    # Add horizontal bar chart
    fig.add_trace(
        go.Bar(
            y=feature_importance['Feature'],
            x=feature_importance['Importance'],
            orientation='h',
            marker=dict(
                color='rgba(0, 0, 255, 0.6)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name="Importance"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=height,
        template="plotly_white",
        margin=dict(l=160, r=40, t=50, b=40),
    )
    
    return fig
