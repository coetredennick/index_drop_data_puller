import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.event_detection import get_all_events, get_event_label
from utils.visualizations import create_recovery_chart, create_technical_indicator_chart
from utils.technical_indicators import get_indicator_explanation

def show_drop_events():
    """
    Display the Drop Events Analysis tab with individual event analysis
    """
    # Check if data and events are available
    if not st.session_state.data is not None or st.session_state.data.empty:
        st.warning("No data available. Please adjust the date range and fetch data.")
        return
    
    # Get all events (both single-day and consecutive)
    all_events = get_all_events()
    
    if not all_events:
        st.warning(f"No drop events found with the current threshold ({st.session_state.drop_threshold}%). Try lowering the threshold.")
        return
    
    # Event selection
    st.markdown("### Select Drop Event")
    
    # Create event labels for dropdown
    event_labels = [get_event_label(event) for event in all_events]
    
    # Get index of previously selected event
    initial_index = 0
    if st.session_state.selected_event is not None:
        # Find the index of the previously selected event
        for i, event in enumerate(all_events):
            if (event['date'] == st.session_state.selected_event['date'] and 
                event['type'] == st.session_state.selected_event['type']):
                initial_index = i
                break
    
    # Create dropdown
    selected_label = st.selectbox(
        "Choose a drop event to analyze:",
        options=event_labels,
        index=initial_index
    )
    
    # Get the selected event
    selected_index = event_labels.index(selected_label)
    selected_event = all_events[selected_index]
    
    # Update session state
    st.session_state.selected_event = selected_event
    
    # Display event details
    st.markdown("### Event Details")
    
    # Create columns for event details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if selected_event['type'] == 'single_day':
            st.metric(
                "Event Date", 
                selected_event['date'].strftime('%Y-%m-%d')
            )
        else:
            st.metric(
                "Event Period", 
                f"{selected_event['start_date'].strftime('%Y-%m-%d')} to {selected_event['date'].strftime('%Y-%m-%d')}"
            )
    
    with col2:
        if selected_event['type'] == 'single_day':
            st.metric(
                "Drop Magnitude", 
                f"{selected_event['drop_pct']:.2f}%",
                delta=None,
                delta_color="inverse"
            )
        else:
            st.metric(
                "Cumulative Drop", 
                f"{selected_event['cumulative_drop']:.2f}%",
                delta=f"{selected_event['num_days']} days",
                delta_color="inverse"
            )
    
    with col3:
        severity = selected_event['severity']
        severity_color = {
            'Severe': 'red',
            'Major': 'orange',
            'Significant': 'yellow',
            'Minor': 'blue'
        }.get(severity, 'gray')
        
        st.markdown(
            f"""
            <div style="border:1px solid {severity_color}; border-radius:5px; padding:10px; text-align:center;">
                <h4 style="margin:0; color:{severity_color};">{severity} Drop</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Show OHLC data for the drop day
    st.markdown("### Price Data")
    
    if selected_event['type'] == 'single_day':
        # Get data for the drop day
        drop_date = selected_event['date']
        if drop_date in st.session_state.data.index:
            drop_day = st.session_state.data.loc[drop_date]
            
            # Create columns for OHLC data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Open", 
                    f"${drop_day['Open']:.2f}"
                )
            
            with col2:
                st.metric(
                    "High", 
                    f"${drop_day['High']:.2f}",
                    delta=f"{((drop_day['High'] / drop_day['Open']) - 1) * 100:.2f}%",
                    delta_color="normal"
                )
            
            with col3:
                st.metric(
                    "Low", 
                    f"${drop_day['Low']:.2f}",
                    delta=f"{((drop_day['Low'] / drop_day['Open']) - 1) * 100:.2f}%",
                    delta_color="normal"
                )
            
            with col4:
                st.metric(
                    "Close", 
                    f"${drop_day['Close']:.2f}",
                    delta=f"{((drop_day['Close'] / drop_day['Open']) - 1) * 100:.2f}%",
                    delta_color="normal"
                )
            
            # Volume information
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Volume' in drop_day and 'Avg_Vol_50' in drop_day:
                    volume_ratio = drop_day['Volume'] / drop_day['Avg_Vol_50']
                    st.metric(
                        "Volume", 
                        f"{drop_day['Volume']:,.0f}",
                        delta=f"{(volume_ratio - 1) * 100:.1f}% vs 50-day avg",
                        delta_color="normal"
                    )
            
            with col2:
                if 'HL_Range' in drop_day:
                    st.metric(
                        "High-Low Range", 
                        f"{drop_day['HL_Range']:.2f}%"
                    )
    else:
        # Consecutive day drop - show table with data for each day
        start_date = selected_event['start_date']
        end_date = selected_event['date']
        
        if start_date in st.session_state.data.index and end_date in st.session_state.data.index:
            period_data = st.session_state.data.loc[start_date:end_date].copy()
            
            # Create a DataFrame with the relevant information
            price_data = pd.DataFrame({
                'Date': period_data.index,
                'Open': period_data['Open'],
                'High': period_data['High'],
                'Low': period_data['Low'],
                'Close': period_data['Close'],
                'Daily Change (%)': period_data['Return'],
                'Volume': period_data['Volume'],
                'Volume vs Avg': period_data['Volume'] / period_data['Avg_Vol_50'] if 'Avg_Vol_50' in period_data.columns else None
            })
            
            # Format the DataFrame
            price_data['Date'] = price_data['Date'].dt.strftime('%Y-%m-%d')
            
            # Function to apply color formatting
            def color_negative_red(val):
                if isinstance(val, (int, float)):
                    if val < 0:
                        return 'color: red'
                return ''
            
            # Apply styling
            styled_price_data = price_data.style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Daily Change (%)': '{:.2f}%',
                'Volume': '{:,.0f}',
                'Volume vs Avg': '{:.2f}x'
            }).applymap(color_negative_red)
            
            # Display the table
            st.dataframe(styled_price_data)
            
            # Calculate and show cumulative metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Starting Price", 
                    f"${period_data['Open'].iloc[0]:.2f}"
                )
            
            with col2:
                st.metric(
                    "Ending Price", 
                    f"${period_data['Close'].iloc[-1]:.2f}"
                )
            
            with col3:
                st.metric(
                    "Total Change", 
                    f"{((period_data['Close'].iloc[-1] / period_data['Open'].iloc[0]) - 1) * 100:.2f}%",
                    delta_color="inverse"
                )
    
    # Show recovery chart
    st.markdown("### Recovery Performance")
    
    # Display forward returns as metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    periods = [
        (col1, '1W', '1 Week'),
        (col2, '1M', '1 Month'),
        (col3, '3M', '3 Months'),
        (col4, '6M', '6 Months'),
        (col5, '1Y', '1 Year'),
        (col6, '3Y', '3 Years')
    ]
    
    for col, period_key, period_label in periods:
        with col:
            return_key = f'fwd_return_{period_key.lower()}'
            if return_key in selected_event and not pd.isna(selected_event[return_key]):
                value = selected_event[return_key]
                
                # Determine color based on value
                if value > 0:
                    delta_color = "normal"  # Green for positive
                else:
                    delta_color = "inverse"  # Red for negative
                
                st.metric(
                    period_label, 
                    f"{value:.2f}%",
                    delta=None,
                    delta_color=delta_color
                )
            else:
                st.metric(
                    period_label, 
                    "N/A"
                )
    
    # Create and display recovery chart
    recovery_fig = create_recovery_chart(st.session_state.data, selected_event)
    st.plotly_chart(recovery_fig, use_container_width=True)
    
    # Show technical indicators
    st.markdown("### Technical Analysis")
    
    # Create cards for technical indicators
    indicators = ['RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Volume_Ratio']
    
    # Create columns for indicators
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for i, indicator in enumerate(indicators):
        with cols[i % 3]:
            if indicator in selected_event and not pd.isna(selected_event[indicator]):
                value = selected_event[indicator]
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
    
    # Create tabs for individual indicator charts
    indicator_tabs = st.tabs([
        "RSI", "Stochastic", "Bollinger Bands", "MACD", "ATR", "Volume"
    ])
    
    # RSI Chart
    with indicator_tabs[0]:
        rsi_fig = create_technical_indicator_chart(
            st.session_state.data, 
            selected_event, 
            'RSI_14', 
            "RSI (14) Around Drop Event"
        )
        st.plotly_chart(rsi_fig, use_container_width=True)
    
    # Stochastic Chart
    with indicator_tabs[1]:
        stoch_fig = create_technical_indicator_chart(
            st.session_state.data, 
            selected_event, 
            'STOCHk_14_3_3', 
            "Stochastic %K (14,3,3) Around Drop Event"
        )
        st.plotly_chart(stoch_fig, use_container_width=True)
    
    # Bollinger Band Position Chart
    with indicator_tabs[2]:
        bb_fig = create_technical_indicator_chart(
            st.session_state.data, 
            selected_event, 
            'BBP_20_2', 
            "Bollinger Band Position (20,2) Around Drop Event"
        )
        st.plotly_chart(bb_fig, use_container_width=True)
    
    # MACD Histogram Chart
    with indicator_tabs[3]:
        macd_fig = create_technical_indicator_chart(
            st.session_state.data, 
            selected_event, 
            'MACDh_12_26_9', 
            "MACD Histogram (12,26,9) Around Drop Event"
        )
        st.plotly_chart(macd_fig, use_container_width=True)
    
    # ATR Chart
    with indicator_tabs[4]:
        atr_fig = create_technical_indicator_chart(
            st.session_state.data, 
            selected_event, 
            'ATR_Pct', 
            "ATR % Around Drop Event"
        )
        st.plotly_chart(atr_fig, use_container_width=True)
    
    # Volume Chart
    with indicator_tabs[5]:
        vol_fig = create_technical_indicator_chart(
            st.session_state.data, 
            selected_event, 
            'Volume_Ratio', 
            "Volume Ratio Around Drop Event"
        )
        st.plotly_chart(vol_fig, use_container_width=True)
