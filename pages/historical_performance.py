import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.event_detection import get_all_events, get_drop_severity
from utils.visualizations import create_price_chart, create_returns_heatmap, create_distribution_histogram

def show_historical_performance():
    """
    Display the Historical Performance tab with aggregate return analysis,
    event distribution, and detailed return database
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
    
    # Display overview metrics
    st.markdown("### S&P 500 Drop Events Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Drop Events", 
            len(all_events),
            help="Total number of detected drop events within the selected date range"
        )
    
    with col2:
        # Count events by severity
        severity_counts = {
            'Severe': 0,
            'Major': 0,
            'Significant': 0,
            'Minor': 0
        }
        
        for event in all_events:
            severity = event['severity']
            severity_counts[severity] += 1
        
        severe_pct = severity_counts['Severe'] / len(all_events) * 100
        st.metric(
            "Severe Drops (>7%)", 
            f"{severity_counts['Severe']} ({severe_pct:.1f}%)",
            help="Number and percentage of severe drop events (>7%)"
        )
    
    with col3:
        major_pct = severity_counts['Major'] / len(all_events) * 100
        st.metric(
            "Major Drops (5-7%)", 
            f"{severity_counts['Major']} ({major_pct:.1f}%)",
            help="Number and percentage of major drop events (5-7%)"
        )
    
    with col4:
        significant_pct = severity_counts['Significant'] / len(all_events) * 100
        st.metric(
            "Significant Drops (3-5%)", 
            f"{severity_counts['Significant']} ({significant_pct:.1f}%)",
            help="Number and percentage of significant drop events (3-5%)"
        )
    
    # Show price chart with drop events marked
    st.markdown("### S&P 500 Price Chart with Drop Events")
    fig_price = create_price_chart(st.session_state.data, all_events)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Show event distribution histogram
    st.markdown("### Distribution of Drop Events")
    fig_dist = create_distribution_histogram(all_events)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Calculate aggregate returns after drops
    st.markdown("### Aggregate Return Analysis After Market Drops")
    
    # Create a DataFrame for aggregate returns
    time_periods = ['1W', '1M', '3M', '6M', '1Y', '3Y']
    
    # Initialize dict to store results
    agg_returns = {
        'Time Period': time_periods,
        'Average Return (%)': [],
        'Median Return (%)': [],
        'Minimum Return (%)': [],
        'Maximum Return (%)': [],
        'Positive Outcomes (%)': []
    }
    
    for period in time_periods:
        period_key = f'fwd_return_{period.lower()}'
        period_returns = [event[period_key] for event in all_events if period_key in event and not pd.isna(event[period_key])]
        
        if period_returns:
            agg_returns['Average Return (%)'].append(np.mean(period_returns))
            agg_returns['Median Return (%)'].append(np.median(period_returns))
            agg_returns['Minimum Return (%)'].append(min(period_returns))
            agg_returns['Maximum Return (%)'].append(max(period_returns))
            positive_pct = sum(1 for r in period_returns if r > 0) / len(period_returns) * 100
            agg_returns['Positive Outcomes (%)'].append(positive_pct)
        else:
            agg_returns['Average Return (%)'].append(None)
            agg_returns['Median Return (%)'].append(None)
            agg_returns['Minimum Return (%)'].append(None)
            agg_returns['Maximum Return (%)'].append(None)
            agg_returns['Positive Outcomes (%)'].append(None)
    
    # Create DataFrame and display
    agg_returns_df = pd.DataFrame(agg_returns)
    
    # Convert to a more readable format with time periods as index
    agg_returns_df = agg_returns_df.set_index('Time Period')
    
    # Function to apply color formatting
    def color_scale(val):
        if pd.isna(val):
            return ''
        
        # For percentage columns
        if isinstance(val, (int, float)):
            if 'Positive Outcomes' in val.name:
                # Green scale for positive outcomes
                color_val = min(1.0, max(0.0, val / 100))
                return f'background-color: rgba(0, 128, 0, {color_val:.2f})'
            elif val > 0:
                # Green for positive returns
                color_val = min(1.0, max(0.0, val / 20))
                return f'background-color: rgba(0, 128, 0, {color_val:.2f})'
            else:
                # Red for negative returns
                color_val = min(1.0, max(0.0, abs(val) / 20))
                return f'background-color: rgba(255, 0, 0, {color_val:.2f})'
        return ''
    
    # Apply styling to the DataFrame
    styled_df = agg_returns_df.style.format('{:.2f}').apply(lambda x: [color_scale(v) for v in x], axis=0)
    
    # Display the table
    st.table(styled_df)
    
    # Calculate average returns by severity
    st.markdown("### Returns by Drop Severity")
    
    # Group events by severity
    severity_groups = {}
    for event in all_events:
        severity = event['severity']
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(event)
    
    # Initialize dict to store results
    severity_returns = {
        'Severity': [],
        'Count': []
    }
    
    # Add columns for each time period
    for period in time_periods:
        severity_returns[f'{period} Avg Return (%)'] = []
        severity_returns[f'{period} Positive (%)'] = []
    
    # Calculate metrics for each severity group
    severity_order = ['Severe', 'Major', 'Significant', 'Minor']
    for severity in severity_order:
        if severity in severity_groups:
            events = severity_groups[severity]
            severity_returns['Severity'].append(severity)
            severity_returns['Count'].append(len(events))
            
            for period in time_periods:
                period_key = f'fwd_return_{period.lower()}'
                period_returns = [event[period_key] for event in events if period_key in event and not pd.isna(event[period_key])]
                
                if period_returns:
                    avg_return = np.mean(period_returns)
                    positive_pct = sum(1 for r in period_returns if r > 0) / len(period_returns) * 100
                    
                    severity_returns[f'{period} Avg Return (%)'].append(avg_return)
                    severity_returns[f'{period} Positive (%)'].append(positive_pct)
                else:
                    severity_returns[f'{period} Avg Return (%)'].append(None)
                    severity_returns[f'{period} Positive (%)'].append(None)
    
    # Create DataFrame and display
    severity_returns_df = pd.DataFrame(severity_returns)
    
    # Convert to a more readable format with severity as index
    severity_returns_df = severity_returns_df.set_index('Severity')
    
    # Apply styling to the DataFrame
    def color_severity_scale(val):
        if pd.isna(val):
            return ''
        
        # For count column
        if 'Count' in val.name:
            return ''
        
        # For percentage columns with 'Positive' in name
        if 'Positive' in val.name:
            # Green scale for positive outcomes
            color_val = min(1.0, max(0.0, val / 100))
            return f'background-color: rgba(0, 128, 0, {color_val:.2f})'
        
        # For return percentage columns
        if val > 0:
            # Green for positive returns
            color_val = min(1.0, max(0.0, val / 20))
            return f'background-color: rgba(0, 128, 0, {color_val:.2f})'
        else:
            # Red for negative returns
            color_val = min(1.0, max(0.0, abs(val) / 20))
            return f'background-color: rgba(255, 0, 0, {color_val:.2f})'
    
    # Apply styling to the DataFrame
    styled_severity_df = severity_returns_df.style.format('{:.2f}').apply(lambda x: [color_severity_scale(v) for v in x], axis=0)
    
    # Display the table
    st.table(styled_severity_df)
    
    # Detailed Return Database
    st.markdown("### Detailed Return Database")
    
    # Prepare data for the heatmap
    events_df = pd.DataFrame([
        {
            'Date': event['date'].strftime('%Y-%m-%d'),
            'Type': 'Single Day' if event['type'] == 'single_day' else f'Consecutive ({event["num_days"]} days)',
            'Drop (%)': event['drop_pct'] if event['type'] == 'single_day' else event['cumulative_drop'],
            'Severity': event['severity'],
            '1W Return (%)': event.get('fwd_return_1w', None),
            '1M Return (%)': event.get('fwd_return_1m', None),
            '3M Return (%)': event.get('fwd_return_3m', None),
            '6M Return (%)': event.get('fwd_return_6m', None),
            '1Y Return (%)': event.get('fwd_return_1y', None),
            '3Y Return (%)': event.get('fwd_return_3y', None)
        }
        for event in all_events
    ])
    
    # Sort by date (newest first)
    events_df = events_df.sort_values('Date', ascending=False)
    
    # Function to apply color formatting for the detailed database
    def color_cell(val):
        if pd.isna(val):
            return 'background-color: #f9f9f9'
        
        if isinstance(val, (int, float)):
            if val > 0:
                # Green for positive returns
                color_val = min(1.0, max(0.1, val / 20))
                return f'background-color: rgba(0, 128, 0, {color_val:.2f})'
            else:
                # Red for negative returns
                color_val = min(1.0, max(0.1, abs(val) / 20))
                return f'background-color: rgba(255, 0, 0, {color_val:.2f})'
        
        # For severity column
        if val == 'Severe':
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
        elif val == 'Major':
            return 'background-color: rgba(255, 0, 0, 0.5); color: white'
        elif val == 'Significant':
            return 'background-color: rgba(255, 0, 0, 0.3)'
        elif val == 'Minor':
            return 'background-color: rgba(255, 0, 0, 0.1)'
            
        return ''
    
    # Apply styling
    styled_events_df = events_df.style.applymap(color_cell)
    
    # Display the table
    st.dataframe(styled_events_df, height=400)
    
    # Add download button for the detailed database
    if not events_df.empty:
        csv = events_df.to_csv(index=False)
        st.download_button(
            label="Download Drop Events Database",
            data=csv,
            file_name=f"sp500_drop_events_{st.session_state.drop_threshold}pct.csv",
            mime="text/csv",
        )
