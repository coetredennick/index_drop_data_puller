import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def create_unified_chart(
    data, 
    drop_events=None, 
    ml_prediction=None, 
    time_period="All Data", 
    show_forecast=False, 
    show_sma=True,
    sma_period=50,
    highlight_drops=True,
    show_rsi=True,
    rsi_period=14,
    show_volume=True,
    height=800
):
    """
    Create a unified visualization chart with price, volume, and RSI in a single figure
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with 'Close', 'Volume', etc.
    drop_events : list, optional
        List of drop events to mark on the chart
    ml_prediction : dict, optional
        ML prediction data including forecast prices
    time_period : str, optional
        Time period to display ("All Data", "10 Years", "5 Years", "1 Year", "YTD")
    show_forecast : bool, optional
        Whether to show the ML forecast
    show_sma : bool, optional
        Whether to show the Simple Moving Average
    sma_period : int, optional
        Period for the Simple Moving Average
    highlight_drops : bool, optional
        Whether to highlight significant drops
    show_rsi : bool, optional
        Whether to show the RSI indicator
    rsi_period : int, optional
        Period for the RSI indicator
    show_volume : bool, optional
        Whether to show the volume chart
    height : int, optional
        Height of the chart in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Unified chart with price, volume, and RSI
    """
    if data is None or data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="S&P 500 Performance Visualization",
            annotations=[dict(
                text="No data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Filter data based on time period
    filtered_data = filter_data_by_time_period(data, time_period)
    
    # Create subplots: main price chart, volume, and RSI
    row_count = 1 + show_volume + show_rsi
    row_heights = [0.6]
    if show_volume:
        row_heights.append(0.2)
    if show_rsi:
        row_heights.append(0.2)
    
    fig = make_subplots(
        rows=row_count, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=[]
    )
    
    # Add main price chart (S&P 500)
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data['Close'],
            mode='lines',
            name='S&P 500 Price',
            line=dict(color='#1E88E5', width=1.5),
            hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Add SMA if requested
    if show_sma:
        sma_name = f'{sma_period}-day SMA'
        if f'SMA_{sma_period}' not in filtered_data.columns:
            filtered_data[f'SMA_{sma_period}'] = filtered_data['Close'].rolling(window=sma_period).mean()
        
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[f'SMA_{sma_period}'],
                mode='lines',
                name=sma_name,
                line=dict(color='#FF9800', width=1, dash='dot'),
                hovertemplate='%{x}<br>' + sma_name + ': $%{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )
    
    # Add ML prediction if available
    if show_forecast and ml_prediction is not None and ml_prediction.get('success', False):
        # Add forecast line
        if 'forecast_dates' in ml_prediction and 'forecast_prices' in ml_prediction:
            # Add the last actual price point to connect the lines
            last_date = filtered_data.index[-1]
            forecast_dates = [last_date] + ml_prediction['forecast_dates']
            forecast_prices = [filtered_data['Close'].iloc[-1]] + ml_prediction['forecast_prices']
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_prices,
                    mode='lines',
                    name='ML Prediction',
                    line=dict(color='#4CAF50', width=1.5),
                    hovertemplate='%{x}<br>Forecast: $%{y:.2f}<extra></extra>',
                ),
                row=1, col=1
            )
            
            # Add confidence intervals if available
            if 'upper_bound' in ml_prediction and 'lower_bound' in ml_prediction:
                # Include the last actual price for upper/lower bounds
                upper_bound = [forecast_prices[0]] + ml_prediction['upper_bound']
                lower_bound = [forecast_prices[0]] + ml_prediction['lower_bound']
                
                # Add upper bound
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                
                # Add lower bound
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(76, 175, 80, 0.1)',
                        fill='tonexty',
                        showlegend=False,
                        hoverinfo='skip',
                        name='Confidence Interval'
                    ),
                    row=1, col=1
                )
    
    # Highlight significant drops if requested
    if highlight_drops and drop_events is not None and len(drop_events) > 0:
        # Filter drop events to the displayed time period
        start_date = filtered_data.index[0]
        end_date = filtered_data.index[-1]
        
        filtered_drops = [
            event for event in drop_events
            if start_date <= pd.to_datetime(event['date']) <= end_date
        ]
        
        # Add markers for drop events
        drop_dates = [pd.to_datetime(event['date']) for event in filtered_drops]
        drop_prices = [filtered_data.loc[pd.to_datetime(event['date']), 'Close'] 
                      if pd.to_datetime(event['date']) in filtered_data.index else None 
                      for event in filtered_drops]
        
        # Remove None values
        valid_indices = [i for i, price in enumerate(drop_prices) if price is not None]
        drop_dates = [drop_dates[i] for i in valid_indices]
        drop_prices = [drop_prices[i] for i in valid_indices]
        filtered_drops = [filtered_drops[i] for i in valid_indices]
        
        if drop_dates and drop_prices:
            # Create hover text for drop events
            hover_texts = []
            for event in filtered_drops:
                if 'consecutive' in event and event['consecutive']:
                    text = f"<b>{event.get('start_date', 'N/A')} to {event.get('end_date', 'N/A')}</b><br>"
                    text += f"Consecutive {event.get('days', 'N/A')} day drop<br>"
                    text += f"Total: {event.get('total_drop', 0.0):.2f}%"
                else:
                    text = f"<b>{event.get('date', 'N/A')}</b><br>"
                    # Check if 'drop' key exists, otherwise try alternate keys or use a default
                    drop_value = event.get('drop')
                    if drop_value is None:
                        drop_value = event.get('total_drop', 0.0)
                    text += f"Single day drop: {drop_value:.2f}%"
                hover_texts.append(text)
            
            fig.add_trace(
                go.Scatter(
                    x=drop_dates,
                    y=drop_prices,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='circle',
                        line=dict(width=1, color='darkred')
                    ),
                    name='Significant Drop Event',
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts,
                ),
                row=1, col=1
            )
    
    # Add volume chart if requested
    current_row = 1
    if show_volume:
        current_row += 1
        
        # Calculate colors for volume bars (green for up days, red for down days)
        colors = ['#4CAF50' if filtered_data['Return'].iloc[i] >= 0 else '#FF5252' 
                 for i in range(len(filtered_data))]
        
        fig.add_trace(
            go.Bar(
                x=filtered_data.index,
                y=filtered_data['Volume'],
                marker=dict(color=colors),
                name='Volume',
                hovertemplate='%{x}<br>Volume: %{y:,.0f}<extra></extra>',
            ),
            row=current_row, col=1
        )
    
    # Add RSI chart if requested
    if show_rsi:
        current_row += 1
        
        # Calculate RSI if not already in the dataframe
        rsi_col = f'RSI_{rsi_period}'
        if rsi_col not in filtered_data.columns:
            # Simple RSI calculation if pandas_ta is not used
            delta = filtered_data['Close'].diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss
            filtered_data[rsi_col] = 100 - (100 / (1 + rs))
        
        # Add RSI trace
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[rsi_col],
                mode='lines',
                name=f'RSI ({rsi_period})',
                line=dict(color='purple', width=1),
                hovertemplate='%{x}<br>RSI: %{y:.2f}<extra></extra>',
            ),
            row=current_row, col=1
        )
        
        # Add RSI reference lines (30 and 70)
        fig.add_shape(
            type='line',
            x0=filtered_data.index[0],
            y0=30,
            x1=filtered_data.index[-1],
            y1=30,
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'),
            row=current_row, col=1
        )
        
        fig.add_shape(
            type='line',
            x0=filtered_data.index[0],
            y0=70,
            x1=filtered_data.index[-1],
            y1=70,
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'),
            row=current_row, col=1
        )
        
        # Add reference background for RSI
        fig.add_shape(
            type='rect',
            x0=filtered_data.index[0],
            y0=0,
            x1=filtered_data.index[-1],
            y1=30,
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(width=0),
            row=current_row, col=1
        )
        
        fig.add_shape(
            type='rect',
            x0=filtered_data.index[0],
            y0=70,
            x1=filtered_data.index[-1],
            y1=100,
            fillcolor='rgba(0,255,0,0.1)',
            line=dict(width=0),
            row=current_row, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="S&P 500 Performance Visualization",
        title_x=0.5,
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    
    # Update y-axis titles for subplots
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    if show_rsi:
        rsi_row = 2 + show_volume
        fig.update_yaxes(title_text=f"RSI ({rsi_period})", row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
    
    # Add slight background color to main chart
    fig.update_layout(
        plot_bgcolor='rgba(240, 242, 246, 0.3)',
        paper_bgcolor='white'
    )
    
    return fig

def filter_data_by_time_period(data, time_period):
    """
    Filter data based on the selected time period
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data
    time_period : str
        Time period to filter ("All Data", "10 Years", "5 Years", "1 Year", "YTD")
        
    Returns:
    --------
    pandas.DataFrame
        Filtered data
    """
    if data is None or data.empty:
        return pd.DataFrame()
    
    today = datetime.today()
    
    if time_period == "10 Years":
        start_date = today - timedelta(days=365 * 10)
        filtered_data = data[data.index >= start_date]
    elif time_period == "5 Years":
        start_date = today - timedelta(days=365 * 5)
        filtered_data = data[data.index >= start_date]
    elif time_period == "1 Year":
        start_date = today - timedelta(days=365)
        filtered_data = data[data.index >= start_date]
    elif time_period == "YTD":
        start_date = datetime(today.year, 1, 1)
        filtered_data = data[data.index >= start_date]
    else:  # "All Data"
        filtered_data = data.copy()
    
    return filtered_data

def create_unified_visualization_ui(data, drop_events=None, consecutive_drop_events=None, ml_model_result=None):
    """
    Create a unified visualization UI with time period selection and chart options
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data
    drop_events : list
        List of single-day drop events
    consecutive_drop_events : list
        List of consecutive-day drop events
    ml_model_result : dict
        Machine learning model result with forecast data
    """
    # Time period selector
    st.write("## S&P 500 Performance Visualization")
    
    # Combine all drop events
    all_drop_events = []
    if drop_events:
        all_drop_events.extend(drop_events)
    if consecutive_drop_events:
        all_drop_events.extend(consecutive_drop_events)
    
    # Create a row of buttons for time period selection
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1.5])
    
    with col1:
        all_data_btn = st.button("All Data", key="all_data_btn", use_container_width=True)
    with col2:
        ten_yr_btn = st.button("10 Years", key="ten_yr_btn", use_container_width=True)
    with col3:
        five_yr_btn = st.button("5 Years", key="five_yr_btn", use_container_width=True)
    with col4:
        one_yr_btn = st.button("1 Year", key="one_yr_btn", use_container_width=True)
    with col5:
        ytd_btn = st.button("YTD", key="ytd_btn", use_container_width=True)
    with col6:
        show_forecast = st.checkbox("Show ML Forecast", value=ml_model_result is not None and ml_model_result.get('success', False))
    
    # Determine selected time period
    if "time_period" not in st.session_state:
        st.session_state.time_period = "All Data"
    
    if all_data_btn:
        st.session_state.time_period = "All Data"
    elif ten_yr_btn:
        st.session_state.time_period = "10 Years"
    elif five_yr_btn:
        st.session_state.time_period = "5 Years"
    elif one_yr_btn:
        st.session_state.time_period = "1 Year"
    elif ytd_btn:
        st.session_state.time_period = "YTD"
    
    # Add small subtitle explaining what is shown
    subtitle = f"Showing {st.session_state.time_period} data with significant drops highlighted in red"
    if show_forecast:
        subtitle += " and ML forecast in green"
    st.caption(subtitle)
    
    # Prepare ML prediction data for chart
    prediction_data = None
    if show_forecast and ml_model_result is not None and ml_model_result.get('success', False):
        prediction_data = ml_model_result.copy()
    
    # Create the unified chart
    chart = create_unified_chart(
        data=data,
        drop_events=all_drop_events,
        ml_prediction=prediction_data,
        time_period=st.session_state.time_period,
        show_forecast=show_forecast,
        show_sma=True,
        sma_period=50,
        highlight_drops=True,
        show_rsi=True,
        show_volume=True,
        height=700
    )
    
    # Display the chart
    st.plotly_chart(chart, use_container_width=True)
    
    # Add a legend/key explanation
    with st.expander("📝 Chart Legend and Explanation"):
        legend_col1, legend_col2 = st.columns(2)
        
        with legend_col1:
            st.markdown("""
            **Chart Elements:**
            - **Blue Line**: S&P 500 Price
            - **Orange Dotted Line**: 50-day Simple Moving Average
            - **Red Circles**: Significant Market Drop Events
            - **Green Line** (if shown): Machine Learning Price Forecast
            - **Green Shaded Area** (if shown): Forecast Confidence Interval
            """)
            
        with legend_col2:
            st.markdown("""
            **Bottom Panels:**
            - **Volume**: Trading volume with green/red indicating up/down days
            - **RSI**: Relative Strength Index (14-day)
              - Above 70 (green zone): Potentially overbought
              - Below 30 (red zone): Potentially oversold
            """)
    
    return chart

def get_ml_prediction_data(ml_result, data, features, days_to_forecast=30):
    """
    Format ML prediction data for the unified chart
    
    Parameters:
    -----------
    ml_result : dict
        Dictionary containing the trained model and performance metrics
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data
    features : list
        List of feature column names used for prediction
    days_to_forecast : int
        Number of days to forecast
        
    Returns:
    --------
    dict
        Formatted prediction data for the unified chart
    """
    from utils.ml_models import predict_returns
    
    if ml_result is None or not ml_result.get('success', False):
        return None
    
    if data is None or data.empty:
        return None
    
    # Get the last available data point
    last_date = data.index[-1]
    last_price = data['Close'].iloc[-1]
    
    # Create a date range for the forecast period
    try:
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
    except Exception as e:
        # Handle date conversion issues
        forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=days_to_forecast)
    
    # Get the ML model prediction
    recent_data = data.tail(1)
    pred_return = predict_returns(ml_result, recent_data, features)
    
    if pred_return is None:
        return None
    
    # Identify which target period the model was trained on
    target_column = ml_result.get('target_column', 'Fwd_Ret_1M')
    target_period = target_column.replace('Fwd_Ret_', '') if 'Fwd_Ret_' in target_column else '1M'
    
    # Map target periods to approximate trading days
    trading_days_map = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '3Y': 756}
    target_days = trading_days_map.get(target_period, 21)  # Default to 1M if unknown
    
    # Scale prediction to daily returns
    daily_return = float(pred_return) / target_days
    
    # Generate forecast prices and confidence intervals
    forecast_prices = []
    upper_bound = []
    lower_bound = []
    
    # Get RMSE for confidence intervals
    rmse = float(ml_result['metrics'].get('rmse_test', 2.0))
    
    # Current price
    current_price = float(last_price)
    
    for i in range(days_to_forecast):
        # Time factor increases variation as we move further into the future
        time_factor = min(1.0, float(i) / (days_to_forecast * 0.3))
        variation_scale = float(rmse * 0.1 * (1 + time_factor))
        
        # Daily return with some variation
        day_return = float(daily_return)
        
        # Calculate next price (compound growth)
        next_price = current_price * (1 + day_return/100)
        forecast_prices.append(float(next_price))
        
        # Calculate confidence intervals (wider as time increases)
        ci_width = float(rmse * (1 + time_factor))
        upper_bound.append(float(next_price * (1 + ci_width/100)))
        lower_bound.append(float(next_price * (1 - ci_width/100)))
        
        # Update current price for next iteration
        current_price = next_price
    
    return {
        'success': True,
        'forecast_dates': forecast_dates.tolist(),
        'forecast_prices': forecast_prices,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound
    }