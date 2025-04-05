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

from utils.data_fetcher import get_latest_sp500_data
from utils.technical_indicators import calculate_technical_indicators, get_indicator_explanation
from utils.visualizations import create_price_chart

def show_current_market():
    """
    Display the Current Market Conditions tab with real-time S&P 500 analysis and forecasting
    """
    
    # Add custom styling for the current market page
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
        
        /* Dashboard card style */
        .dashboard-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Metric styling */
        div[data-testid="stMetric"] {
            background-color: white;
            padding: 0.7rem;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Metric label styling */
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }
        
        /* Improve plot styling */
        .stPlotlyChart {
            margin-bottom: 1rem;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1rem;
        }
        
        /* Indicator card styling */
        .indicator-card {
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            background-color: white;
        }
        
        /* Indicator title */
        .indicator-title {
            font-weight: 500;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
            color: #1E4A7B;
        }
        
        /* Indicator value */
        .indicator-value {
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Indicator description */
        .indicator-desc {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("### Current S&P 500 Market Conditions")
    
    # Fetch the latest data
    with st.spinner("Fetching latest market data..."):
        current_data = get_latest_sp500_data()
    
    data_available = current_data is not None and not current_data.empty
    
    if not data_available:
        st.error("Failed to fetch current market data. Please try again later.")
        current_data = pd.DataFrame()  # Empty DataFrame to avoid NoneType errors
    
    # Calculate technical indicators if not already present
    if 'RSI_14' not in current_data.columns:
        current_data = calculate_technical_indicators(current_data)
    
    # Get the most recent data point
    latest_data = current_data.iloc[-1]
    prev_data = current_data.iloc[-2] if len(current_data) > 1 else None
    
    # Current price and daily change
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Daily change calculation
        if prev_data is not None:
            daily_change = (latest_data['Close'] - prev_data['Close']) / prev_data['Close'] * 100
            daily_change_color = "normal" if daily_change >= 0 else "inverse"
        else:
            daily_change = 0
            daily_change_color = "normal"
        
        st.metric(
            "Current S&P 500 Price", 
            f"${latest_data['Close']:.2f}",
            delta=f"{daily_change:.2f}%" if prev_data is not None else None,
            delta_color=daily_change_color
        )
    
    with col2:
        # Year-to-date calculation
        start_of_year = datetime(datetime.now().year, 1, 1)
        ytd_start_price = None
        
        # Find the first trading day of the year in our data
        for date, row in current_data.iterrows():
            if date >= start_of_year:
                ytd_start_price = row['Close']
                break
                
        if ytd_start_price is not None:
            ytd_change = (latest_data['Close'] - ytd_start_price) / ytd_start_price * 100
            ytd_change_color = "normal" if ytd_change >= 0 else "inverse"
        else:
            ytd_change = 0
            ytd_change_color = "normal"
        
        st.metric(
            "Year-to-Date Performance", 
            f"{ytd_change:.2f}%" if ytd_start_price is not None else "N/A",
            delta=None,
            delta_color=ytd_change_color
        )
    
    with col3:
        # Last updated timestamp
        last_update = latest_data.name.strftime('%Y-%m-%d %H:%M:%S')
        
        st.metric(
            "Last Updated", 
            last_update
        )
    
    # Price chart
    st.markdown("### Recent Price Movement")
    
    # Create 30-day price chart
    days_to_show = min(30, len(current_data))
    recent_data = current_data.iloc[-days_to_show:]
    
    fig = create_price_chart(recent_data, title="S&P 500 - Last 30 Days")
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical analysis dashboard
    st.markdown("### Technical Indicators Dashboard")
    
    # Create cards for technical indicators
    indicators = ['RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Volume_Ratio']
    
    # Create two rows of indicators, 3 per row
    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    
    all_cols = row1_cols + row2_cols
    
    for i, indicator in enumerate(indicators):
        with all_cols[i]:
            if indicator in latest_data and not pd.isna(latest_data[indicator]):
                value = latest_data[indicator]
                explanation = get_indicator_explanation(indicator, value)
                
                # Determine card color based on status
                if explanation['status'] == 'bullish':
                    card_color = 'rgba(0, 128, 0, 0.1)'
                    border_color = 'rgba(0, 128, 0, 0.5)'
                    text_color = 'green'
                elif explanation['status'] == 'bearish':
                    card_color = 'rgba(255, 0, 0, 0.1)'
                    border_color = 'rgba(255, 0, 0, 0.5)'
                    text_color = 'darkred'
                else:
                    card_color = 'rgba(128, 128, 128, 0.1)'
                    border_color = 'rgba(128, 128, 128, 0.5)'
                    text_color = 'gray'
                
                # Create card with styling
                st.markdown(
                    f"""
                    <div style="border:1px solid {border_color}; border-radius:5px; padding:10px; background-color:{card_color}; margin-bottom:10px;">
                        <h4 style="margin:0; color:{text_color};">{explanation['title']}</h4>
                        <div style="font-size:24px; font-weight:bold; margin:5px 0;">{value:.2f}</div>
                        <p style="margin:0; font-size:12px;">{explanation['explanation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # No data available
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; background-color:#f9f9f9; margin-bottom:10px;">
                        <h4 style="margin:0; color:#666;">{indicator}</h4>
                        <div style="font-size:24px; font-weight:bold; margin:5px 0;">N/A</div>
                        <p style="margin:0; font-size:12px;">No data available for this indicator.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Moving average analysis
    st.markdown("### Moving Average Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 50-day SMA
        if 'SMA_50' in latest_data and not pd.isna(latest_data['SMA_50']):
            sma50 = latest_data['SMA_50']
            price_to_sma50 = (latest_data['Close'] / sma50 - 1) * 100
            
            delta_color = "normal" if price_to_sma50 >= 0 else "inverse"
            
            st.metric(
                "50-Day SMA", 
                f"${sma50:.2f}",
                delta=f"{price_to_sma50:.2f}% vs price",
                delta_color=delta_color
            )
        else:
            st.metric("50-Day SMA", "N/A")
    
    with col2:
        # 200-day SMA
        if 'SMA_200' in latest_data and not pd.isna(latest_data['SMA_200']):
            sma200 = latest_data['SMA_200']
            price_to_sma200 = (latest_data['Close'] / sma200 - 1) * 100
            
            delta_color = "normal" if price_to_sma200 >= 0 else "inverse"
            
            st.metric(
                "200-Day SMA", 
                f"${sma200:.2f}",
                delta=f"{price_to_sma200:.2f}% vs price",
                delta_color=delta_color
            )
        else:
            st.metric("200-Day SMA", "N/A")
    
    with col3:
        # Golden/Death Cross
        if 'SMA_50' in latest_data and 'SMA_200' in latest_data:
            if not pd.isna(latest_data['SMA_50']) and not pd.isna(latest_data['SMA_200']):
                sma50 = latest_data['SMA_50']
                sma200 = latest_data['SMA_200']
                
                # Check for golden or death cross
                if sma50 > sma200:
                    # Check if a golden cross occurred recently (within last 10 days)
                    recent_golden_cross = False
                    for i in range(min(10, len(current_data)-1)):
                        idx = -2-i
                        if idx < -len(current_data):
                            break
                        prev_row = current_data.iloc[idx]
                        if 'SMA_50' in prev_row and 'SMA_200' in prev_row:
                            if prev_row['SMA_50'] <= prev_row['SMA_200']:
                                recent_golden_cross = True
                                break
                    
                    if recent_golden_cross:
                        st.markdown(
                            """
                            <div style="border:1px solid gold; border-radius:5px; padding:10px; background-color:rgba(255, 215, 0, 0.1); text-align:center;">
                                <h4 style="margin:0; color:darkgoldenrod;">RECENT GOLDEN CROSS!</h4>
                                <p style="margin:5px 0; font-size:12px;">50-day SMA crossed above 200-day SMA (bullish)</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="border:1px solid green; border-radius:5px; padding:10px; background-color:rgba(0, 128, 0, 0.1); text-align:center;">
                                <h4 style="margin:0; color:green;">BULLISH TREND</h4>
                                <p style="margin:5px 0; font-size:12px;">50-day SMA above 200-day SMA</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    # Check if a death cross occurred recently (within last 10 days)
                    recent_death_cross = False
                    for i in range(min(10, len(current_data)-1)):
                        idx = -2-i
                        if idx < -len(current_data):
                            break
                        prev_row = current_data.iloc[idx]
                        if 'SMA_50' in prev_row and 'SMA_200' in prev_row:
                            if prev_row['SMA_50'] >= prev_row['SMA_200']:
                                recent_death_cross = True
                                break
                    
                    if recent_death_cross:
                        st.markdown(
                            """
                            <div style="border:1px solid darkred; border-radius:5px; padding:10px; background-color:rgba(139, 0, 0, 0.1); text-align:center;">
                                <h4 style="margin:0; color:darkred;">RECENT DEATH CROSS!</h4>
                                <p style="margin:5px 0; font-size:12px;">50-day SMA crossed below 200-day SMA (bearish)</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="border:1px solid red; border-radius:5px; padding:10px; background-color:rgba(255, 0, 0, 0.1); text-align:center;">
                                <h4 style="margin:0; color:red;">BEARISH TREND</h4>
                                <p style="margin:5px 0; font-size:12px;">50-day SMA below 200-day SMA</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.markdown(
                """
                <div style="border:1px solid gray; border-radius:5px; padding:10px; background-color:#f9f9f9; text-align:center;">
                    <h4 style="margin:0; color:gray;">MOVING AVERAGE CROSSOVER</h4>
                    <p style="margin:5px 0; font-size:12px;">Data not available</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Volatility analysis
    st.markdown("### Volatility Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 30-day volatility (annualized)
        days_for_vol = min(30, len(current_data))
        recent_data_vol = current_data.iloc[-days_for_vol:]
        
        if 'Return' in recent_data_vol.columns:
            volatility_30d = recent_data_vol['Return'].std() * np.sqrt(252) # Annualized
            
            # Compare to historical average (if we have enough data)
            if len(current_data) > 252:  # Approximately 1 year of trading days
                historical_vol = current_data['Return'].std() * np.sqrt(252)
                vol_ratio = volatility_30d / historical_vol
                
                delta_val = f"{(vol_ratio - 1) * 100:.1f}% vs historical"
                delta_color = "normal" if vol_ratio < 1 else "inverse"  # Lower volatility is better
            else:
                delta_val = None
                delta_color = "normal"
            
            st.metric(
                "30-Day Volatility (Annualized)", 
                f"{volatility_30d:.2f}%",
                delta=delta_val,
                delta_color=delta_color
            )
        else:
            st.metric("30-Day Volatility (Annualized)", "N/A")
    
    with col2:
        # Maximum drawdown over the past 30 days
        if len(recent_data_vol) > 1:
            # Calculate rolling maximum
            rolling_max = recent_data_vol['Close'].cummax()
            
            # Calculate drawdown
            drawdown = (recent_data_vol['Close'] / rolling_max - 1) * 100
            
            # Get maximum drawdown
            max_drawdown = drawdown.min()
            
            st.metric(
                "30-Day Maximum Drawdown", 
                f"{max_drawdown:.2f}%",
                delta=None,
                delta_color="inverse"  # Drawdown is always negative
            )
        else:
            st.metric("30-Day Maximum Drawdown", "N/A")
    
    # Simple Price Forecast
    st.markdown("### S&P 500 Price Forecast")
    
    # Create a simple forecasting model based on moving averages and recent trend
    if len(current_data) >= 30:
        # Define different forecasting methods
        forecasting_method = st.selectbox(
            "Forecasting Method",
            ["Moving Average Trend", "Recent Momentum", "Linear Regression"],
            index=0
        )
        
        # Number of days to forecast
        days_to_forecast = st.slider("Days to Forecast", min_value=7, max_value=90, value=30, step=7)
        
        # Last actual price
        last_date = current_data.index[-1]
        last_price = current_data['Close'].iloc[-1]
        
        # Generate forecast dates
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
        
        # Moving Average Trend method
        if forecasting_method == "Moving Average Trend":
            # Use the ratio between short and long term moving averages to project future prices
            # Calculate short and long moving averages
            short_ma_days = 10
            long_ma_days = 30
            
            if len(current_data) > long_ma_days:
                short_ma = current_data['Close'].rolling(window=short_ma_days).mean().iloc[-1]
                long_ma = current_data['Close'].rolling(window=long_ma_days).mean().iloc[-1]
                
                # Calculate daily growth rate based on MA ratio
                ma_ratio = short_ma / long_ma
                daily_growth = (ma_ratio - 1) * 5 / long_ma_days  # Scale to appropriate daily change
                
                # Generate forecast prices with momentum decay
                forecast_prices = []
                for i in range(days_to_forecast):
                    if i == 0:
                        next_price = last_price * (1 + daily_growth)
                    else:
                        # Momentum decays over time
                        decay_factor = max(0.1, 1 - (i / days_to_forecast))
                        next_price = forecast_prices[-1] * (1 + daily_growth * decay_factor)
                    forecast_prices.append(next_price)
            else:
                st.warning("Not enough historical data for Moving Average Trend forecast.")
                forecast_prices = [last_price] * days_to_forecast  # Flat forecast as fallback
        
        # Recent Momentum method
        elif forecasting_method == "Recent Momentum":
            # Calculate recent momentum (average daily return)
            recent_days = 5
            recent_returns = current_data['Return'].iloc[-recent_days:]
            avg_daily_return = recent_returns.mean()
            
            # Generate forecast with momentum decay
            forecast_prices = []
            for i in range(days_to_forecast):
                if i == 0:
                    next_price = last_price * (1 + avg_daily_return/100)
                else:
                    # Momentum decays over time
                    decay_factor = max(0.1, 1 - (i / days_to_forecast))
                    next_price = forecast_prices[-1] * (1 + (avg_daily_return/100) * decay_factor)
                forecast_prices.append(next_price)
        
        # Linear Regression method
        else:  # Linear Regression
            from sklearn.linear_model import LinearRegression
            # Use the global numpy import instead of re-importing
            
            # Use last 30 days for regression
            regression_days = min(30, len(current_data))
            regression_data = current_data.iloc[-regression_days:]
            
            # Prepare data for regression
            X = np.array(range(regression_days)).reshape(-1, 1)
            y = regression_data['Close'].values
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict future values
            X_future = np.array(range(regression_days, regression_days + days_to_forecast)).reshape(-1, 1)
            forecast_prices = model.predict(X_future)
        
        # Create the plot
        # Use the global plotly.graph_objects import (already imported as go at the top)
        
        fig = go.Figure()
        
        # Add historical data
        historical_data = current_data.iloc[-30:]  # Show last 30 days
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_prices,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Add marker for last actual price
        fig.add_trace(
            go.Scatter(
                x=[last_date],
                y=[last_price],
                mode='markers',
                marker=dict(color='black', size=8, symbol='circle'),
                name='Latest Price'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"S&P 500 {days_to_forecast}-Day Price Forecast ({forecasting_method})",
            xaxis_title="Date",
            yaxis_title="S&P 500 Price ($)",
            height=500,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add disclaimer
        st.markdown("""
        **Disclaimer:** This forecast is based on simple technical analysis and historical patterns. 
        It should not be used as the sole basis for investment decisions. The actual market movement 
        may differ significantly due to unforeseen events, economic factors, or market sentiment.
        """)
    else:
        st.warning("Not enough historical data available for forecasting.")
    
    # Volume analysis
    st.markdown("### Volume Analysis")
    
    # Create volume chart for last 10 days
    days_for_vol_chart = min(10, len(current_data))
    volume_data = current_data.iloc[-days_for_vol_chart:].copy()
    
    # Calculate average volume if available
    if 'Avg_Vol_50' in volume_data.columns:
        avg_volume = volume_data['Avg_Vol_50'].mean()
    else:
        avg_volume = volume_data['Volume'].mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add volume bars
    for i, (date, row) in enumerate(volume_data.iterrows()):
        # Determine color based on price change
        if 'Return' in row and not pd.isna(row['Return']):
            color = 'rgba(0, 128, 0, 0.7)' if row['Return'] >= 0 else 'rgba(255, 0, 0, 0.7)'
        else:
            color = 'rgba(128, 128, 128, 0.7)'
        
        # Add bar
        fig.add_trace(
            go.Bar(
                x=[date],
                y=[row['Volume']],
                marker_color=color,
                name=date.strftime('%Y-%m-%d'),
                text=f"{row['Volume'] / 1000000:.1f}M",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.0f}<extra></extra>"
            )
        )
    
    # Add average volume line
    fig.add_trace(
        go.Scatter(
            x=volume_data.index,
            y=[avg_volume] * len(volume_data),
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='50-Day Average'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="10-Day Volume Analysis",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=400,
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Format y-axis to show millions
    fig.update_yaxes(tickformat=".1s")
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Volume' in latest_data and 'Avg_Vol_50' in latest_data:
            latest_volume = latest_data['Volume']
            avg_volume_50d = latest_data['Avg_Vol_50']
            volume_ratio = latest_volume / avg_volume_50d
            
            st.metric(
                "Latest Volume", 
                f"{latest_volume / 1000000:.1f}M",
                delta=f"{(volume_ratio - 1) * 100:.1f}% vs 50-day avg",
                delta_color="normal"  # Higher volume is generally better for confirmation
            )
        else:
            st.metric("Latest Volume", "N/A")
    
    with col2:
        # Volume trend (increasing or decreasing)
        if 'Volume' in current_data.columns and len(current_data) >= 10:
            recent_volumes = current_data['Volume'].iloc[-10:].tolist()
            
            # Simple trend calculation
            volume_trend = sum(1 for i in range(1, len(recent_volumes)) if recent_volumes[i] > recent_volumes[i-1])
            volume_trend_pct = volume_trend / (len(recent_volumes) - 1) * 100
            
            if volume_trend_pct > 60:
                trend_text = "Increasing"
                delta_color = "normal"
            elif volume_trend_pct < 40:
                trend_text = "Decreasing"
                delta_color = "inverse"
            else:
                trend_text = "Stable"
                delta_color = "normal"
            
            st.metric(
                "Volume Trend (10 Days)", 
                trend_text,
                delta=f"{volume_trend_pct:.1f}% of days increasing",
                delta_color=delta_color
            )
        else:
            st.metric("Volume Trend (10 Days)", "N/A")
