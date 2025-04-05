import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_features(data, focus_on_drops=True, drop_threshold=-3.0):
    """
    Prepare features for machine learning models
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with technical indicators
    focus_on_drops : bool, optional
        Whether to focus specifically on market drop events
    drop_threshold : float, optional
        Percentage threshold for considering a day as a market drop
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features for ML models, and list of feature names
    """
    # Create a copy of the data
    df = data.copy()
    
    # Check for required columns
    required_cols = ['RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Return']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # If missing columns, print info for debugging and return empty data
        print(f"Warning: Missing columns for ML model: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame(), []
    
    # Select features for the model - only include those that exist
    base_features = [
        'RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct'
    ]
    
    # Filter to only include columns that exist
    features = [f for f in base_features if f in df.columns]
    
    # Add Volume Ratio if available
    if 'Volume_Ratio' in df.columns:
        features.append('Volume_Ratio')
    elif 'Volume' in df.columns and 'Avg_Vol_50' in df.columns:
        # Calculate volume ratio if not already present
        df['Volume_Ratio'] = df['Volume'] / df['Avg_Vol_50']
        features.append('Volume_Ratio')
    
    # Add price-based features
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']
        features.append('SMA_50_200_Ratio')
    
    if 'EMA_20' in df.columns and 'Close' in df.columns:
        df['EMA_20_Close_Ratio'] = df['EMA_20'] / df['Close']
        features.append('EMA_20_Close_Ratio')
    
    # Add lagged returns
    if 'Return' in df.columns:
        for lag in [1, 5, 10, 20]:
            df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
            features.append(f'Return_Lag_{lag}')
        
        # Calculate rolling volatility
        df['Volatility_20'] = df['Return'].rolling(window=20).std()
        features.append('Volatility_20')
    
    # Add features specifically focused on market drops
    # Identify consecutive drop days
    if 'Return' in df.columns:
        # Mark significant drop days
        df['Is_Drop_Day'] = df['Return'] <= drop_threshold
        
        # Count consecutive drop days
        df['Drop_Streak'] = 0
        current_streak = 0
        streak_values = []
        
        # First collect all streak values
        for i in range(len(df)):
            if df['Is_Drop_Day'].iloc[i]:
                current_streak += 1
            else:
                current_streak = 0
            streak_values.append(current_streak)
        
        # Then set them all at once to avoid the SettingWithCopyWarning
        df.loc[:, 'Drop_Streak'] = streak_values
        
        features.append('Drop_Streak')
        
        # Calculate magnitude of drops (cumulative over streaks)
        df['Cumulative_Drop'] = 0
        cumulative = 0
        cumulative_values = []
        
        # First collect all values
        for i in range(len(df)):
            if df['Is_Drop_Day'].iloc[i]:
                cumulative += df['Return'].iloc[i]
            else:
                cumulative = 0
            cumulative_values.append(cumulative)
        
        # Then set them all at once to avoid the SettingWithCopyWarning
        df.loc[:, 'Cumulative_Drop'] = cumulative_values
        
        features.append('Cumulative_Drop')
        
        # Add recent market behavior features
        df['Recent_Max_Drop'] = df['Return'].rolling(window=10).min()
        df['Recent_Volatility'] = df['Return'].rolling(window=10).std()
        
        features.extend(['Recent_Max_Drop', 'Recent_Volatility'])
    
    if not features:
        print("Warning: No valid features available for ML model.")
        return pd.DataFrame(), []
    
    # Focus specifically on market drops if requested
    if focus_on_drops and 'Return' in df.columns:
        print(f"Focusing on market drops (threshold: {drop_threshold}%)")
        # Either a drop day or the day after a drop or in a streak
        drop_condition = (
            (df['Return'] <= drop_threshold) | 
            (df['Return'].shift(1) <= drop_threshold) | 
            (df['Drop_Streak'] > 0)
        )
        filtered_df = df[drop_condition].copy()
        
        # If filtering results in too few data points, use a more relaxed threshold
        if len(filtered_df) < 30:  # Minimum number for reliable training
            print(f"Warning: Too few data points ({len(filtered_df)}) with threshold {drop_threshold}%. Using a more relaxed threshold.")
            relaxed_threshold = max(drop_threshold * 0.5, -1.0)  # More relaxed threshold
            drop_condition = (
                (df['Return'] <= relaxed_threshold) | 
                (df['Return'].shift(1) <= relaxed_threshold) | 
                (df['Drop_Streak'] > 0)
            )
            filtered_df = df[drop_condition].copy()
            print(f"Relaxed threshold to {relaxed_threshold}%, got {len(filtered_df)} market drop events")
        
        df = filtered_df
        print(f"Filtered to {len(df)} market drop events for training")
    
    # Drop NaN values in feature columns
    df_clean = df.dropna(subset=features)
    
    # Check if we have enough data left
    if len(df_clean) < 50:  # Arbitrary threshold - need enough data for training
        print(f"Warning: Not enough data after removing NaN values: {len(df_clean)} rows")
        return pd.DataFrame(), []
    
    # Debug: Check target columns
    for period in ['1W', '1M', '3M', '6M', '1Y', '3Y']:
        col_name = f'Fwd_Ret_{period}'
        if col_name in df_clean.columns:
            non_nan = df_clean[col_name].count()
            percent = (non_nan / len(df_clean)) * 100
            print(f"Column {col_name}: {non_nan}/{len(df_clean)} non-NaN values ({percent:.1f}%)")
    
    return df_clean, features

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

def create_forecast_chart(model_result, data, features, days_to_forecast=30, title="S&P 500 Forecast", height=500):
    """
    Create a chart showing the forecasted S&P 500 price based on the trained model
    
    Parameters:
    -----------
    model_result : dict
        Dictionary containing the trained model and performance metrics
    data : pandas.DataFrame
        DataFrame containing historical S&P 500 data
    features : list
        List of feature column names used for prediction
    days_to_forecast : int, optional
        Number of days to forecast into the future
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Forecast chart
    """
    if not model_result['success']:
        # No model available
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No model available for forecasting",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Get the last available data point
    last_date = data.index[-1]
    last_price = data['Close'].iloc[-1]
    
    # Create a date range for the forecast period
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
    
    # Use the model to make predictions for each forecast day
    forecast_prices = []
    current_data = data.copy()
    
    for i in range(days_to_forecast):
        # Ensure we have all required features for prediction
        if all(feature in current_data.columns for feature in features):
            # Get prediction for the next day's return
            pred_return = predict_returns(model_result, current_data, features)
            
            if pred_return is not None:
                # Calculate predicted price based on return
                if i == 0:
                    # First prediction based on last actual price
                    pred_price = last_price * (1 + pred_return/100)
                else:
                    # Subsequent predictions based on previous prediction
                    pred_price = forecast_prices[-1] * (1 + pred_return/100)
                
                forecast_prices.append(pred_price)
                
                # Add this prediction to the data for the next iteration
                # Create a new row for the forecasted day
                new_row = pd.DataFrame({
                    'Open': pred_price,
                    'High': pred_price * 1.005,  # Approximate
                    'Low': pred_price * 0.995,   # Approximate
                    'Close': pred_price,
                    'Volume': current_data['Volume'].mean(),  # Use mean volume
                    'Return': pred_return
                }, index=[forecast_dates[i]])
                
                # Add technical indicators to new row (simplified)
                if 'RSI_14' in current_data.columns:
                    new_row['RSI_14'] = current_data['RSI_14'].iloc[-1]
                if 'MACD_12_26_9' in current_data.columns:
                    new_row['MACD_12_26_9'] = current_data['MACD_12_26_9'].iloc[-1]
                
                # Append to the data
                current_data = pd.concat([current_data, new_row])
            else:
                # If prediction fails, use last price
                forecast_prices.append(forecast_prices[-1] if forecast_prices else last_price)
        else:
            # Missing features, use last price
            forecast_prices.append(forecast_prices[-1] if forecast_prices else last_price)
    
    # Create the figure
    fig = go.Figure()
    
    # Add historical data (last 60 days)
    historical_data = data.iloc[-60:]
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='rgba(0, 0, 255, 0.8)', width=2)
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            mode='lines',
            name='Forecast',
            line=dict(color='rgba(255, 0, 0, 0.8)', width=2, dash='dash')
        )
    )
    
    # Calculate confidence interval (using model's RMSE)
    rmse = model_result['metrics']['rmse_test']
    upper_bound = [price * (1 + rmse/100) for price in forecast_prices]
    lower_bound = [price * (1 - rmse/100) for price in forecast_prices]
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            showlegend=False,
            name='Lower Bound'
        )
    )
    
    # Add a marker for the last actual price
    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[last_price],
            mode='markers',
            marker=dict(color='black', size=8, symbol='circle'),
            name='Latest Price'
        )
    )
    
    # Format the layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="S&P 500 Price ($)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
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
