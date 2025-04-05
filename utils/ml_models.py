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
            streak_values.append(int(current_streak))
        
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
                cumulative += float(df['Return'].iloc[i])
            else:
                cumulative = 0
            cumulative_values.append(float(cumulative))
        
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
        'model_type': model_type,
        'target_column': target_column,
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

def create_forecast_chart(model_result, data, features, days_to_forecast=30, title="S&P 500 ML Forecast", height=500):
    """
    Create a chart showing the forecasted S&P 500 price based on the trained ML model
    
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
    # Create the figure
    fig = go.Figure()
    
    # Error handling: Check if model_result is None or not successful
    if model_result is None or not model_result.get('success', False):
        # No model available
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No ML model available for forecasting. Train a model first.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Error handling: Check if data is empty
    if data is None or data.empty or 'Close' not in data.columns:
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="Insufficient historical data for ML forecasting",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Error handling: Check if all required features are available
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        fig.update_layout(
            title=title,
            annotations=[dict(
                text=f"Missing ML features: {', '.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}",
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
    try:
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
    except Exception as e:
        # Handle date conversion issues
        forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=days_to_forecast)
    
    # Find YTD data for better visualization
    try:
        current_year = last_date.year
        start_of_year = pd.Timestamp(year=current_year, month=1, day=1)
        ytd_data = data[data.index >= start_of_year]
        if len(ytd_data) < 5:  # If not enough YTD data, use last 60 days
            ytd_data = data.tail(60)
    except:
        # Fallback to last 60 days if year calculation fails
        ytd_data = data.tail(60)
    
    # Use the model to make predictions using ML model
    try:
        # Get the most recent market data for prediction
        recent_data = data.tail(1)
        
        # Check if we have all required features
        if len(recent_data) > 0 and all(feature in recent_data.columns for feature in features):
            # Get the ML model prediction 
            pred_return = predict_returns(model_result, recent_data, features)
            
            if pred_return is not None:
                # Identify which target period the model was trained on (1W, 1M, 3M, etc.)
                # This helps scale our predictions appropriately
                target_column = model_result.get('target_column', 'Fwd_Ret_1M')
                target_period = target_column.replace('Fwd_Ret_', '') if 'Fwd_Ret_' in target_column else '1M'
                
                # Map target periods to approximate trading days for proper scaling
                trading_days_map = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '3Y': 756}
                target_days = trading_days_map.get(target_period, 21)  # Default to 1M if unknown
                
                # Scale prediction to daily returns (approximate)
                daily_return = pred_return / target_days
                
                # Generate forecast prices with compound growth using ML prediction
                forecast_prices = [last_price]
                
                for i in range(days_to_forecast):
                    # Create more realistic variations in daily returns
                    # - Less variation in near term (more confidence)
                    # - More variation further out (less confidence)
                    # - Base variation on model's RMSE (accuracy measure)
                    rmse = float(model_result['metrics'].get('rmse_test', 2.0))
                    time_factor = min(1.0, i / (days_to_forecast * 0.3))  # Ramps up variation over time
                    variation_scale = float(rmse * 0.1 * (1 + time_factor))  # Scale variation based on time and model accuracy
                    
                    # Add some realistic market behavior variation
                    variation = float(np.random.normal(0, variation_scale))
                    
                    # Daily return with appropriate variation
                    day_return = float(daily_return + variation)
                    
                    # Calculate next price with compounding
                    next_price = forecast_prices[-1] * (1 + day_return/100)
                    forecast_prices.append(next_price)
                
                # Remove the initial price (which is the last actual price)
                forecast_prices = forecast_prices[1:]
                
                # Calculate robust confidence intervals based on model accuracy
                rmse = model_result['metrics'].get('rmse_test', 2.0)  # RMSE from model evaluation
                ci_factor = 1.96  # 95% confidence interval (statistical standard)
                
                # Generate more realistic confidence intervals that widen with time
                upper_prices = []
                lower_prices = []
                
                for i, price in enumerate(forecast_prices):
                    # Uncertainty grows with square root of time (finance theory)
                    days_out = int(i + 1)
                    time_factor = float(np.sqrt(days_out / 5))  # Scale based on days out
                    uncertainty = float((rmse * ci_factor * time_factor) / 100)
                    
                    upper_prices.append(price * (1 + uncertainty))
                    lower_prices.append(price * (1 - uncertainty))
                
                # Add YTD historical data for context
                fig.add_trace(
                    go.Scatter(
                        x=ytd_data.index,
                        y=ytd_data['Close'],
                        mode='lines',
                        name='YTD Historical',
                        line=dict(color='rgba(0, 0, 255, 0.8)', width=2)
                    )
                )
                
                # Add ML-based forecast line
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_prices,
                        mode='lines',
                        name=f'ML Forecast',
                        line=dict(color='rgba(255, 0, 0, 0.8)', width=2, dash='dash')
                    )
                )
                
                # Add confidence intervals - fill between upper and lower bounds
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=upper_prices,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=lower_prices,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='95% Confidence',
                        hoverinfo='skip'
                    )
                )
                
                # Mark the last actual price point
                fig.add_trace(
                    go.Scatter(
                        x=[last_date],
                        y=[last_price],
                        mode='markers',
                        name='Latest Price',
                        marker=dict(
                            color='black',
                            size=10,
                            symbol='circle'
                        )
                    )
                )
                
                # Calculate key forecast statistics
                end_forecast_price = forecast_prices[-1]
                forecast_change_pct = ((end_forecast_price / last_price) - 1) * 100
                forecast_change_direction = "increase" if forecast_change_pct > 0 else "decrease"
                
                # Create informative annotations
                annotations = [
                    dict(
                        x=forecast_dates[-1],
                        y=end_forecast_price,
                        xref="x",
                        yref="y",
                        text=f"${end_forecast_price:.2f} ({forecast_change_pct:+.1f}%)",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        ax=40,
                        ay=-40,
                        font=dict(
                            color="red" if forecast_change_pct < 0 else "green",
                            size=12
                        )
                    )
                ]
                
                # Update layout with enhanced styling
                fig.update_layout(
                    title=f"{title} (Based on {target_period} Model)",
                    xaxis_title='Date',
                    yaxis_title='S&P 500 Price ($)',
                    height=height,
                    hovermode='x unified',
                    annotations=annotations,
                    yaxis=dict(
                        tickprefix='$',
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                
                # Add range slider and selector buttons for better navigation
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
                
                # Add footer annotation with model details
                model_type = model_result.get('model_type', 'Random Forest')
                model_metrics = f"Model Accuracy: RMSE={rmse:.2f}% | R²={model_result['metrics'].get('r2_test', 0):.2f}"
                
                fig.add_annotation(
                    text=f"ML Model: {model_type} trained on {target_period} returns | {model_metrics}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    align="center"
                )
                
                return fig
                
    except Exception as e:
        # If any error occurs during forecasting, log it for debugging
        import traceback
        print(f"ML Forecasting error: {str(e)}")
        print(traceback.format_exc())
    
    # If we reach here, something went wrong in the ML forecasting
    # Create a fallback chart with just historical data
    historical_days = min(60, len(data))
    if historical_days > 0:
        historical_data = data.iloc[-historical_days:]
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='rgba(0, 0, 255, 0.8)', width=2)
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
            title=title + " (ML Model Not Available)",
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
            annotations=[dict(
                text="ML forecasting failed - please train a model first",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="red")
            )]
        )
    else:
        # No historical data available
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No data available for ML forecasting",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
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
