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
    
    # Add some custom CSS for this page's elements
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
        
        /* Improve table readability */
        table {
            border-collapse: collapse;
            width: 100%;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        th {
            background-color: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }
        
        /* Plot margins */
        .stPlotlyChart {
            margin-bottom: 1rem;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    # Check if data and events are available
    data_available = st.session_state.data is not None and not st.session_state.data.empty
    
    if not data_available:
        st.warning("No data available. Please adjust the date range and fetch data.")
    else:
        # Get all events (both single-day and consecutive)
        all_events = get_all_events()
        
        if not all_events:
            st.warning(f"No drop events found with the current threshold ({st.session_state.drop_threshold}%). Try lowering the threshold.")
            all_events = []  # Empty list to avoid None references later
    
    # Display overview metrics
    st.markdown("### S&P 500 Drop Events Analysis")
    
    # Create cards that highlight key aggregate data 
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Drop Events", 
            len(all_events),
            delta=None
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
            delta=None
        )
    
    with col3:
        major_pct = severity_counts['Major'] / len(all_events) * 100
        st.metric(
            "Major Drops (5-7%)", 
            f"{severity_counts['Major']} ({major_pct:.1f}%)",
            delta=None
        )
    
    with col4:
        significant_pct = severity_counts['Significant'] / len(all_events) * 100
        st.metric(
            "Significant Drops (3-5%)", 
            f"{severity_counts['Significant']} ({significant_pct:.1f}%)",
            delta=None
        )
        

    
    # Show price chart with drop events marked
    st.markdown("### S&P 500 Price Chart with Drop Events")
    fig_price = create_price_chart(st.session_state.data, all_events)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Calculate aggregate returns after drops for use in detailed DB
    time_periods = ['1W', '1M', '3M', '6M', '1Y', '3Y']
    
    # Calculate metrics for later use
    # Create a DataFrame for calculating aggregate metrics
    agg_metrics = {}
    severity_metrics = {}
    
    # Compute overall aggregates
    for period in time_periods:
        period_key = f'fwd_return_{period.lower()}'
        period_returns = [event[period_key] for event in all_events if period_key in event and not pd.isna(event[period_key])]
        
        if period_returns:
            agg_metrics[period] = {
                'avg': np.mean(period_returns),
                'med': np.median(period_returns),
                'min': min(period_returns),
                'max': max(period_returns),
                'pos_pct': sum(1 for r in period_returns if r > 0) / len(period_returns) * 100
            }
        else:
            agg_metrics[period] = {
                'avg': None, 'med': None, 'min': None, 'max': None, 'pos_pct': None
            }
    
    # Compute by severity
    severity_groups = {}
    for event in all_events:
        severity = event['severity']
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(event)
    
    severity_order = ['Severe', 'Major', 'Significant', 'Minor']
    for severity in severity_order:
        if severity in severity_groups:
            events = severity_groups[severity]
            severity_metrics[severity] = {'count': len(events)}
            
            for period in time_periods:
                period_key = f'fwd_return_{period.lower()}'
                period_returns = [event[period_key] for event in events if period_key in event and not pd.isna(event[period_key])]
                
                if period_returns:
                    avg_return = np.mean(period_returns)
                    positive_pct = sum(1 for r in period_returns if r > 0) / len(period_returns) * 100
                    severity_metrics[severity][f'{period}_avg'] = avg_return
                    severity_metrics[severity][f'{period}_pos'] = positive_pct
                else:
                    severity_metrics[severity][f'{period}_avg'] = None
                    severity_metrics[severity][f'{period}_pos'] = None
    
    # Key Performance Metrics (moved to directly above the table)
    st.markdown("### Key Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Calculate average return metrics for key timeframes
    if all_events and len(all_events) > 0:
        # Get average 1 month return
        avg_1m = np.mean([event.get('fwd_return_1m', 0) for event in all_events 
                          if 'fwd_return_1m' in event and not pd.isna(event['fwd_return_1m'])])
        # Percentage of positive 1 month returns
        positive_1m = sum(1 for event in all_events 
                         if 'fwd_return_1m' in event and not pd.isna(event['fwd_return_1m']) and event['fwd_return_1m'] > 0)
        positive_1m_pct = (positive_1m / len(all_events)) * 100 if len(all_events) > 0 else 0
        
        # Get average 1 year return
        avg_1y = np.mean([event.get('fwd_return_1y', 0) for event in all_events 
                          if 'fwd_return_1y' in event and not pd.isna(event['fwd_return_1y'])])
        # Percentage of positive 1 year returns
        positive_1y = sum(1 for event in all_events 
                         if 'fwd_return_1y' in event and not pd.isna(event['fwd_return_1y']) and event['fwd_return_1y'] > 0)
        positive_1y_pct = (positive_1y / len(all_events)) * 100 if len(all_events) > 0 else 0
        
        # Calculate average drop size
        avg_drop = np.mean([abs(event['drop_pct']) if event['type'] == 'single_day' else abs(event['cumulative_drop']) 
                           for event in all_events])
    else:
        avg_1m = 0
        positive_1m_pct = 0
        avg_1y = 0
        positive_1y_pct = 0
        avg_drop = 0
    
    with metrics_col1:
        st.metric(
            "Avg. Drop Size", 
            f"{avg_drop:.1f}%",
            delta=None,
        )
    
    with metrics_col2:
        st.metric(
            "Avg. 1-Month Return", 
            f"{avg_1m:.1f}%",
            delta=f"{positive_1m_pct:.0f}% positive" if positive_1m_pct > 50 else f"only {positive_1m_pct:.0f}% positive",
            delta_color="normal" if positive_1m_pct > 50 else "off"
        )
    
    with metrics_col3:
        st.metric(
            "Avg. 1-Year Return", 
            f"{avg_1y:.1f}%",
            delta=f"{positive_1y_pct:.0f}% positive" if positive_1y_pct > 50 else f"only {positive_1y_pct:.0f}% positive",
            delta_color="normal" if positive_1y_pct > 50 else "off"
        )
    
    with metrics_col4:
        # Calc best and worst month
        if all_events and len(all_events) > 0:
            month_returns = [event.get('fwd_return_1m', 0) for event in all_events 
                           if 'fwd_return_1m' in event and not pd.isna(event['fwd_return_1m'])]
            if month_returns:
                best_month = max(month_returns)
                worst_month = min(month_returns)
                st.metric(
                    "Best vs Worst 1M", 
                    f"+{best_month:.1f}% / {worst_month:.1f}%",
                    delta=f"Range: {best_month - worst_month:.1f}%",
                    delta_color="off"
                )
            else:
                st.metric("Best vs Worst 1M", "N/A", delta=None)
                
    # Detailed Return Database (now combined with aggregates)
    st.markdown("### Market Drop Return Database with Aggregates")
    
    # Create a simple database table with totals at the top
    # Prepare individual event data first
    event_rows = []
    for event in all_events:
        row = {
            'Date': event['date'].strftime('%Y-%m-%d'),
            'Type': 'Single Day' if event['type'] == 'single_day' else f'Consecutive ({event["num_days"]} days)',
            'Drop (%)': event['drop_pct'] if event['type'] == 'single_day' else event['cumulative_drop'],
            'Severity': event['severity'],
            '1W (%)': event.get('fwd_return_1w', None),
            '1M (%)': event.get('fwd_return_1m', None),
            '3M (%)': event.get('fwd_return_3m', None),
            '6M (%)': event.get('fwd_return_6m', None),
            '1Y (%)': event.get('fwd_return_1y', None),
            '3Y (%)': event.get('fwd_return_3y', None)
        }
        event_rows.append(row)
    
    # Create DataFrame for events
    events_df = pd.DataFrame(event_rows)
    
    # Sort by date (newest first)
    if len(events_df) > 0:
        events_df = events_df.sort_values('Date', ascending=False)
        
        # Add a column for the total/average return across periods
        events_df['Total Avg (%)'] = events_df[['1W (%)', '1M (%)', '3M (%)', '6M (%)', '1Y (%)', '3Y (%)']].mean(axis=1)
        
        # Add a totals row at the bottom
        totals_row = {
            'Date': 'TOTALS',
            'Type': f'{len(events_df)} Events',
            'Drop (%)': events_df['Drop (%)'].mean(),
            'Severity': 'All Types',
        }
        
        # Add totals for each return period
        for col in ['1W (%)', '1M (%)', '3M (%)', '6M (%)', '1Y (%)', '3Y (%)', 'Total Avg (%)']:
            totals_row[col] = events_df[col].mean()
        
        # Add totals row at the bottom
        events_df = pd.concat([events_df, pd.DataFrame([totals_row])], ignore_index=True)
    
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
        elif val == 'AVERAGE':
            return 'background-color: rgba(0, 0, 128, 0.7); color: white; font-weight: bold'
        elif val == 'MEDIAN':
            return 'background-color: rgba(0, 0, 128, 0.5); color: white; font-weight: bold'
        elif val == 'MIN':
            return 'background-color: rgba(0, 0, 128, 0.3); color: white; font-weight: bold'
        elif val == 'MAX':
            return 'background-color: rgba(0, 0, 128, 0.3); color: white; font-weight: bold'
        elif val == 'POSITIVE %':
            return 'background-color: rgba(0, 128, 0, 0.7); color: white; font-weight: bold'
        
        # For totals row
        if val == 'TOTALS' or val == 'All Types':
            return 'background-color: #f0f4f8; color: #0056b3; font-weight: bold'
            
        return ''
    
    # Create format dictionary for the detailed table
    format_dict = {}
    for col in events_df.columns:
        if '%' in col:  # Format percentage columns
            # Make sure we only format numeric values
            format_dict[col] = lambda x: '{:.1f}%'.format(x) if isinstance(x, (int, float)) else x
    
    # Apply styling with formatting and smaller text using map (replaces deprecated applymap)
    # Let's define custom CSS directly for the HTML table to ensure styling is preserved
    # Create a custom HTML with our styling to ensure the right columns get formatted
    
    # First format the data values with our formatter
    formatted_df = events_df.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:{fmt}}" if pd.notna(x) else "")
    
    # Apply color formatting
    html_content = "<div style='overflow-x: auto;'>"
    html_content += "<table class='returns-table' style='width:100%; border-collapse: collapse;'>"
    
    # Add header row
    html_content += "<tr>"
    for col in formatted_df.columns:
        html_content += f"<th style='font-size:11px; padding:4px 8px; text-align:center; background-color:#e9ecef; border-bottom:1px solid #adb5bd; white-space:nowrap;'>{col}</th>"
    html_content += "</tr>"
    
    # Add data rows
    row_count = len(formatted_df)
    for i, (_, row) in enumerate(formatted_df.iterrows()):
        is_last_row = i == row_count - 1
        html_content += "<tr>"
        
        for j, (col, val) in enumerate(row.items()):
            # Determine if this is the Total column (index 9)
            is_total_col = j == 9  # 10th column (0-indexed)
            
            # Set the cell style based on position and value
            cell_style = "font-size:10px; padding:2px 5px; white-space:nowrap; "
            
            # Add special formatting for the Total column
            if is_total_col:
                cell_style += "border-left:2px solid #333; background-color:#f0f4f8; font-weight:900; color:#0056b3; text-shadow:0 0 0.2px #0056b3; "
            
            # Add special formatting for the totals row
            if is_last_row:
                cell_style += "border-top:2px solid #333; border-bottom:2px solid #333; background-color:#f0f4f8; font-weight:900; font-size:11px; color:#0056b3; text-shadow:0 0 0.2px #0056b3; "
            
            # Add extra emphasis for the intersection of totals row and column
            if is_last_row and is_total_col:
                cell_style = "font-size:11px; padding:2px 5px; white-space:nowrap; border-top:2px solid #333; border-bottom:2px solid #333; border-left:2px solid #333; background-color:#e6f0ff; font-weight:900; color:#004494; text-shadow:0 0 0.5px #004494; "
            
            # Add color based on value (if it's a number)
            try:
                num_val = float(val.replace('%', '').replace(',', '')) if isinstance(val, str) else val
                if pd.notna(num_val) and not is_last_row:  # Skip coloring for the totals row (already styled)
                    if num_val < 0:
                        cell_style += "color:#d60000; "  # Red for negative
                    elif num_val > 0:
                        cell_style += "color:#008800; "  # Green for positive
            except (ValueError, AttributeError):
                pass  # Not a number or empty, keep default styling
            
            html_content += f"<td style='{cell_style}'>{val}</td>"
        
        html_content += "</tr>"
    
    html_content += "</table></div>"
    
    # Display the custom HTML table
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Add download button for the detailed database
    if not events_df.empty:
        csv = events_df.to_csv(index=False)
        st.download_button(
            label="Download Drop Events Database",
            data=csv,
            file_name=f"sp500_drop_events_{st.session_state.drop_threshold}pct.csv",
            mime="text/csv",
        )
