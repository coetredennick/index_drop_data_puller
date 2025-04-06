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
    
    # Add custom styling for this page
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
        
        /* Style for technical indicators */
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Style for tab content */
        div[data-testid="stTabContent"] {
            background-color: white;
            border-radius: 0 5px 5px 5px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #f8f9fa;
        }
        
        /* Improve plot styling */
        .stPlotlyChart {
            margin-bottom: 1rem;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1rem;
        }
        
        /* Selectbox styling */
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
        
        /* Card styling for technical indicators */
        div[data-baseweb="tab-panel"] {
            padding: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    # Check if data and events are available
    data_available = st.session_state.data is not None and not st.session_state.data.empty
    events_available = False
    
    if not data_available:
        st.warning("No data available. Please adjust the date range and fetch data.")
    else:
        # Add event type filter
        event_type_options = {
            "All Events": "all",
            "Single-Day Drops": "single_day",
            "Consecutive Drops": "consecutive"
        }
        
        event_filter = st.selectbox(
            "Event Type Filter:",
            options=list(event_type_options.keys()),
            index=0,
            key="drop_events_type_filter"
        )
        
        # Get the selected event type value
        selected_event_type = event_type_options[event_filter]
        
        # Store selected event type in session state for use across the app
        st.session_state.current_event_type_filter = selected_event_type
        
        # Get events based on the selected filter
        all_events = get_all_events(event_type=selected_event_type)
        
        if not all_events:
            if selected_event_type == "all":
                st.warning(f"No drop events found with the current threshold ({st.session_state.drop_threshold}%). Try lowering the threshold.")
            elif selected_event_type == "single_day":
                st.warning(f"No single-day drop events found with the current threshold ({st.session_state.drop_threshold}%). Try lowering the threshold.")
            elif selected_event_type == "consecutive":
                st.warning(f"No consecutive drop events found with the current threshold ({st.session_state.drop_threshold}%) and consecutive days setting ({st.session_state.consecutive_days} days). Try adjusting these parameters.")
            all_events = []  # Empty list to avoid None references later
        else:
            events_available = True
    
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
    
    # Custom key for the selectbox to store its state
    dropdown_key = "drop_event_selection_dropdown"
    
    # Initialize the dropdown state if it doesn't exist
    if dropdown_key not in st.session_state:
        st.session_state[dropdown_key] = event_labels[initial_index]
    
    # Function to handle selection change
    def handle_selection_change():
        selected_label = st.session_state[dropdown_key]
        selected_index = event_labels.index(selected_label)
        st.session_state.selected_event = all_events[selected_index]
    
    # Create dropdown outside of a form (no need for the form anymore)
    selected_label = st.selectbox(
        "Choose a drop event to analyze:",
        options=event_labels,
        index=event_labels.index(st.session_state[dropdown_key]),
        key=dropdown_key,
        on_change=handle_selection_change
    )
    
    # No need for a submit button - changes are applied immediately
    
    # Get the currently selected event from session state
    selected_event = st.session_state.selected_event
    
    # Check if selected_event is None (could happen on first load)
    if selected_event is None and len(all_events) > 0:
        # Initialize with the first event
        selected_event = all_events[0]
        st.session_state.selected_event = selected_event
    
    # Only continue if we have a valid selected event
    if selected_event is not None:
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
                    f"{selected_event['drop_pct']:.1f}%",
                    delta=None,
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "Cumulative Drop", 
                    f"{selected_event['cumulative_drop']:.1f}%",
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
                    delta=f"{((drop_day['High'] / drop_day['Open']) - 1) * 100:.1f}%",
                    delta_color="normal"
                )
            
            with col3:
                st.metric(
                    "Low", 
                    f"${drop_day['Low']:.2f}",
                    delta=f"{((drop_day['Low'] / drop_day['Open']) - 1) * 100:.1f}%",
                    delta_color="normal"
                )
            
            with col4:
                st.metric(
                    "Close", 
                    f"${drop_day['Close']:.2f}",
                    delta=f"{((drop_day['Close'] / drop_day['Open']) - 1) * 100:.1f}%",
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
                        f"{drop_day['HL_Range']:.1f}%"
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
                
                # Add a total row at the bottom
                total_row = pd.DataFrame({
                    'Date': ['Total'],
                    'Open': [period_data['Open'].iloc[0]],
                    'High': [period_data['High'].max()],
                    'Low': [period_data['Low'].min()],
                    'Close': [period_data['Close'].iloc[-1]],
                    'Daily Change (%)': [((period_data['Close'].iloc[-1] / period_data['Open'].iloc[0]) - 1) * 100],
                    'Volume': [period_data['Volume'].sum()],
                    'Volume vs Avg': [period_data['Volume'].sum() / (period_data['Avg_Vol_50'].mean() * len(period_data)) if 'Avg_Vol_50' in period_data.columns else None]
                })
                
                # Append the total row
                price_data = pd.concat([price_data, total_row], ignore_index=True)
                
                # Format the Date column - handle both datetime and string types
                for i in range(len(price_data)):
                    if isinstance(price_data.loc[i, 'Date'], pd.Timestamp):
                        price_data.loc[i, 'Date'] = price_data.loc[i, 'Date'].strftime('%Y-%m-%d')
                
                # Function to apply color formatting
                def color_negative_red(val, props=''):
                    if isinstance(val, (int, float)) and props.startswith('Daily Change'):
                        if val < 0:
                            return 'color: red'
                    return ''
                
                # Apply styling with smaller font and decimal formatting
                styled_price_data = price_data.style.format({
                    'Open': '${:.2f}',
                    'High': '${:.2f}',
                    'Low': '${:.2f}',
                    'Close': '${:.2f}',
                    'Daily Change (%)': '{:.1f}%',
                    'Volume': '{:,.0f}',
                    'Volume vs Avg': '{:.1f}x'
                })
                
                # Add custom CSS for smaller font and more compact layout with distinct totals
                styled_price_data = styled_price_data.set_table_styles([
                    {'selector': 'td', 'props': [('font-size', '10px'), ('padding', '2px 5px'), ('white-space', 'nowrap')]},
                    {'selector': 'th', 'props': [
                        ('font-size', '10px'), 
                        ('padding', '2px 5px'), 
                        ('white-space', 'nowrap'),
                        ('background-color', '#e9ecef'),  # Slightly darker background for headers
                        ('border-bottom', '1px solid #adb5bd'),  # Border under headers
                        ('text-align', 'center')  # Center-align headers
                    ]},
                    
                    # Make the "Daily Change (%)" column more distinct (total column)
                    {'selector': 'td:nth-child(6), th:nth-child(6)', 'props': [
                        ('border-left', '2px solid #333'),
                        ('border-right', '2px solid #333'),
                        ('background-color', '#f0f4f8'),  # Light blue-gray background
                        ('font-weight', '900'),          # Extra bold text
                        ('color', '#0056b3'),            # Blue text
                        ('text-shadow', '0 0 0.2px #0056b3')  # Text shadow for emphasis
                    ]},
                    
                    # Make the last row (totals) more distinct
                    {'selector': 'tr:last-child td', 'props': [
                        ('border-top', '2px solid #333'),
                        ('border-bottom', '2px solid #333'),
                        ('background-color', '#f0f4f8'),  # Light blue-gray background
                        ('font-weight', '900'),           # Extra bold text
                        ('font-size', '11px'),            # Slightly larger font
                        ('color', '#0056b3'),             # Blue text
                        ('text-shadow', '0 0 0.2px #0056b3')  # Text shadow for emphasis
                    ]},
                    
                    # Make the intersection of totals row and column even more emphasized
                    {'selector': 'tr:last-child td:nth-child(6)', 'props': [
                        ('background-color', '#e6f0ff'),  # Slightly different background
                        ('font-weight', '900'),           # Extra bold text
                        ('font-size', '11px'),            # Slightly larger font
                        ('color', '#004494'),             # Darker blue for the intersection
                        ('text-shadow', '0 0 0.5px #004494')  # Stronger text shadow
                    ]}
                ])
                
                # Apply color formatting using map (replacing deprecated applymap)
                styled_price_data = styled_price_data.map(color_negative_red)
                
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
                        f"{((period_data['Close'].iloc[-1] / period_data['Open'].iloc[0]) - 1) * 100:.1f}%",
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
                        f"{value:.1f}%",
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
                            <div style="font-size:24px; font-weight:bold; margin:5px 0;">{value:.1f}</div>
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
