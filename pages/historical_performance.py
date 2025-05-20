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
    
    # Initialize all_events to empty list as default
    all_events = []
    
    if not data_available:
        st.warning("No data available. Please adjust the date range and fetch data.")
    else:
        # Add event type filter
        event_type_options = {
            "All Events": "all",
            "Single-Day Drops": "single_day",
            "Consecutive Drops": "consecutive"
        }
        
        # Check which market is active and create a unique key for each market
        active_index = st.session_state.active_index if 'active_index' in st.session_state else 'sp500'
        
        event_filter = st.selectbox(
            "Event Type Filter:",
            options=list(event_type_options.keys()),
            index=0,
            key=f"historical_event_type_filter_{active_index}"
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
        
        total_events = len(all_events) if all_events else 0
        
        for event in all_events:
            severity = event['severity']
            severity_counts[severity] += 1
        
        # Avoid division by zero
        if total_events > 0:
            severe_pct = severity_counts['Severe'] / total_events * 100
            st.metric(
                "Severe Drops (>7%)", 
                f"{severity_counts['Severe']} ({severe_pct:.1f}%)",
                delta=None
            )
        else:
            st.metric(
                "Severe Drops (>7%)", 
                "0 (0.0%)",
                delta=None
            )
    
    with col3:
        if total_events > 0:
            major_pct = severity_counts['Major'] / total_events * 100
            st.metric(
                "Major Drops (5-7%)", 
                f"{severity_counts['Major']} ({major_pct:.1f}%)",
                delta=None
            )
        else:
            st.metric(
                "Major Drops (5-7%)", 
                "0 (0.0%)",
                delta=None
            )
    
    with col4:
        if total_events > 0:
            significant_pct = severity_counts['Significant'] / total_events * 100
            st.metric(
                "Significant Drops (3-5%)", 
                f"{severity_counts['Significant']} ({significant_pct:.1f}%)",
                delta=None
            )
        else:
            st.metric(
                "Significant Drops (3-5%)", 
                "0 (0.0%)",
                delta=None
            )
        

    
    # Show price chart with drop events marked
    st.markdown("### S&P 500 Price Chart with Drop Events")
    if st.session_state.data is not None and not st.session_state.data.empty:
        fig_price = create_price_chart(st.session_state.data, all_events)
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("Cannot create price chart. No data available. Please try adjusting the date range and fetching data again.")
    
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
        # For consecutive drops, create a date range string with all dates
        if event['type'] == 'consecutive':
            # Format: Start date → End date (with all dates in between)
            start_date = pd.Timestamp(event['start_date']) if not isinstance(event['start_date'], pd.Timestamp) else event['start_date']
            end_date = pd.Timestamp(event['date']) if not isinstance(event['date'], pd.Timestamp) else event['date']
            
            # Generate all dates for the period (including non-trading days)
            date_range = pd.date_range(start=start_date, end=end_date)
            date_list = [d.strftime('%Y-%m-%d') for d in date_range]
            date_str = f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}\n({', '.join(date_list)})"
        else:
            # For single day events, just use the date
            date_str = event['date'].strftime('%Y-%m-%d')
            
        # For consecutive drops, format the drop percentage as the sum of the daily drops
        if event['type'] == 'consecutive':
            # Display the detailed daily drops
            daily_drops = event.get('daily_drops', [])
            if daily_drops:
                daily_drop_str = f"{event['cumulative_drop']:.2f}% ({', '.join([f'{d:.2f}%' for d in daily_drops])})"
            else:
                daily_drop_str = f"{event['cumulative_drop']:.2f}%"
        else:
            # For single day events, just show the drop percentage
            daily_drop_str = f"{event['drop_pct']:.2f}%"
            
        # Get rate of decline metrics - now including drawdown from peak
        decline_rate = event.get('decline_rate_per_day', 0)  # Rate within the window/day
        decline_duration = event.get('decline_duration', 1)  # 1-day minimum
        
        # Get peak-based metrics
        peak_to_end_rate = event.get('peak_to_end_rate', decline_rate)
        drawdown_from_peak = event.get('drawdown_from_peak_pct', event.get('drop_pct', 0))
        days_from_peak = event.get('days_since_peak', decline_duration)
        
        # For consecutive drops, add information about acceleration
        if event['type'] == 'consecutive':
            max_daily_decline = event.get('max_daily_decline', 0)
            decline_acceleration = event.get('decline_acceleration', 0)
            # Format acceleration with sign
            acceleration_str = f"{decline_acceleration:+.2f}%" if abs(decline_acceleration) > 0.01 else "0.00%"
            
            # Show both the window metrics and the drawdown from peak metrics
            window_metrics = f"Window: {decline_rate:.2f}%/day over {decline_duration}d"
            peak_metrics = f"From Peak: {peak_to_end_rate:.2f}%/day over {days_from_peak}d"
            detail_metrics = f"Max: {max_daily_decline:.2f}% | Accel: {acceleration_str}"
            
            decline_metrics = f"{window_metrics} | {peak_metrics} | {detail_metrics}"
        else:
            # For single day drops, check if the drawdown from peak is significantly different
            if abs(drawdown_from_peak - event.get('drop_pct', 0)) > 0.5 and days_from_peak > 1:
                # Show both single day and drawdown from peak metrics
                decline_metrics = f"Daily: {decline_rate:.2f}%/day | From Peak: {peak_to_end_rate:.2f}%/day over {days_from_peak}d"
            else:
                # Just show the single-day metrics
                decline_metrics = f"{decline_rate:.2f}%/day | Duration: 1d"
            
        # Get technical indicator data from the event date
        event_date = event['date']
        vix_value = None
        volume_value = None
        rsi_value = None
        
        # Try to get the VIX, Volume and RSI values for this date
        try:
            if event_date in st.session_state.data.index:
                row_data = st.session_state.data.loc[event_date]
                vix_value = row_data.get('VIX_Close', None)
                volume_value = row_data.get('Volume', None)
                rsi_value = row_data.get('RSI_14', None)
        except Exception as e:
            # If data isn't available, just leave as None
            pass
            
        # Format technical indicators
        vix_str = f"{vix_value:.2f}" if vix_value is not None and not pd.isna(vix_value) else "N/A"
        volume_str = f"{volume_value:,.0f}" if volume_value is not None and not pd.isna(volume_value) else "N/A"
        rsi_str = f"{rsi_value:.1f}" if rsi_value is not None and not pd.isna(rsi_value) else "N/A"
            
        row = {
            'Date': date_str,
            'Type': 'Single Day' if event['type'] == 'single_day' else f'Consecutive ({event["num_days"]} days)',
            'Drop (%)': event['drop_pct'] if event['type'] == 'single_day' else event['cumulative_drop'],
            'Daily Drops': daily_drop_str,
            'Severity': event['severity'],
            'Decline Rate': decline_metrics.replace('From Peak: ', '').replace('Daily: ', '').replace('Window: ', ''),  # Shortened format
            'VIX': vix_str,
            'Volume': volume_str,
            'RSI': rsi_str,
            '1D (%)': event.get('fwd_return_1d', None),
            '2D (%)': event.get('fwd_return_2d', None),
            '3D (%)': event.get('fwd_return_3d', None),
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
        events_df['Total Avg (%)'] = events_df[['1D (%)', '2D (%)', '3D (%)', '1W (%)', '1M (%)', '3M (%)', '6M (%)', '1Y (%)', '3Y (%)']].mean(axis=1)
        
        # Add a totals row at the bottom
        totals_row = {
            'Date': 'TOTALS',
            'Type': f'{len(events_df)} Events',
            'Drop (%)': events_df['Drop (%)'].mean(),
            'Severity': 'All Types',
        }
        
        # Add the Daily Drops column for the TOTALS row (leaving it empty since it can't be meaningfully aggregated)
        totals_row['Daily Drops'] = 'N/A'
        
        # Add VIX, Volume, and RSI averages to the totals row
        try:
            # Calculate averages of available technical indicators
            vix_avg = events_df[events_df['VIX'].apply(lambda x: x != 'N/A')]['VIX'].apply(lambda x: float(x)).mean()
            volume_avg = events_df[events_df['Volume'].apply(lambda x: x != 'N/A')]['Volume'].apply(lambda x: float(x.replace(',', ''))).mean()
            rsi_avg = events_df[events_df['RSI'].apply(lambda x: x != 'N/A')]['RSI'].apply(lambda x: float(x)).mean()
            
            # Format them
            totals_row['VIX'] = f"{vix_avg:.2f}" if not pd.isna(vix_avg) else "N/A"
            totals_row['Volume'] = f"{volume_avg:,.0f}" if not pd.isna(volume_avg) else "N/A"
            totals_row['RSI'] = f"{rsi_avg:.1f}" if not pd.isna(rsi_avg) else "N/A"
        except Exception as e:
            # If there's any error in calculating, use N/A
            totals_row['VIX'] = "N/A"
            totals_row['Volume'] = "N/A"
            totals_row['RSI'] = "N/A"
        
        # Add the Decline Rate column for the TOTALS row
        # Calculate average rate and duration across all events
        avg_rate = events_df[events_df['Decline Rate'].notna()]['Drop (%)'].abs().mean() / events_df[events_df['Decline Rate'].notna()]['Type'].apply(
            lambda x: int(x.split('(')[1].split(' ')[0]) if 'Consecutive' in x else 1
        ).mean()
        avg_duration = events_df[events_df['Decline Rate'].notna()]['Type'].apply(
            lambda x: int(x.split('(')[1].split(' ')[0]) if 'Consecutive' in x else 1
        ).mean()
        
        totals_row['Decline Rate'] = f"Avg: {avg_rate:.2f}%/d"
        
        # Add totals for each return period
        for col in ['1D (%)', '2D (%)', '3D (%)', '1W (%)', '1M (%)', '3M (%)', '6M (%)', '1Y (%)', '3Y (%)', 'Total Avg (%)']:
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
    
    # First format the data values with simpler, more robust approach
    formatted_df = events_df.copy()
    
    # Format all columns directly without using format_dict
    for col in formatted_df.columns:
        if '%' in col and col != 'Daily Drops':  # Format percentage columns, except for the detailed daily drops
            # Format as percentage with one decimal place - check for string values
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
            )
        elif col == 'Date' or col == 'Type' or col == 'Severity' or col == 'Daily Drops':
            # Keep these columns as is (string columns)
            continue
        else:
            # Format other numeric columns with commas for thousands - check for string values
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
            )
    
    # Apply color formatting
    html_content = "<div style='overflow-x: auto;'>"
    html_content += "<table class='returns-table' style='width:100%; border-collapse: collapse;'>"
    
    # Add header row
    html_content += "<tr>"
    for col in formatted_df.columns:
        html_content += f"<th style='font-size:11px; padding:4px 8px; text-align:center; background-color:#e9ecef; border-bottom:1px solid #adb5bd; white-space:nowrap;'>{col}</th>"
    html_content += "</tr>"
    
    # Column names are now processed automatically
    
    # Now we know the column index for Total Avg
    # Get the index of the "Total Avg" column
    total_avg_col_index = None
    for idx, col_name in enumerate(formatted_df.columns):
        if "Total" in col_name and "Avg" in col_name:
            total_avg_col_index = idx
            break
    
    # Add data rows
    row_count = len(formatted_df)
    for i, (_, row) in enumerate(formatted_df.iterrows()):
        is_last_row = i == row_count - 1
        html_content += "<tr>"
        
        for j, (col, val) in enumerate(row.items()):
            # Determine if this is the Total column using the discovered index
            is_total_col = total_avg_col_index is not None and j == total_avg_col_index
            
            # Set the cell style based on position and value
            cell_style = "font-size:10px; padding:2px 5px; white-space:nowrap; "
            
            # Add background color heatmap based on value (if it's a number) for all cells except the total column
            try:
                col_name = col.strip()
                num_val = float(val.replace('%', '').replace(',', '')) if isinstance(val, str) else val
                
                # Process only if we have a valid number and it's not the totals row or the totals column
                if pd.notna(num_val) and not is_last_row and not is_total_col:
                    # Default rgb values
                    r, g, b = 255, 255, 255
                    applied_color = False  # Flag to track if we've applied a color
                    
                    # Technical indicators: VIX, RSI, Volume with green-yellow-red gradient
                    if col_name == 'VIX':
                        # VIX normalization - typically ranges from 10-80
                        # High VIX values indicate extreme volatility/fear
                        # We want high VIX (extreme fear) to be deep red
                        
                        # Apply a more exponential scaling to increase contrast
                        # This will make higher VIX values more red more quickly
                        # First make sure we have a non-negative value to avoid complex numbers
                        vix_base = max(0.0, (num_val - 10) / 40.0)
                        vix_norm = vix_base ** 1.5  # Now safe to raise to power
                        
                        # Create common green-yellow-red gradient
                        # Reverse the norm_val calculation so 1.0 = highest VIX (most extreme)
                        norm_val = vix_norm  # Higher VIX = higher norm_val = more red
                        
                        # Calculate colors based on normalized value
                        if norm_val < 0.5:
                            # Green to yellow gradient (0-0.5)
                            r = int(60 + 195 * norm_val * 2)  # 60 -> 255
                            g = int(200 + 55 * norm_val * 2)  # 200 -> 255
                            b = int(90 * (1 - norm_val * 2))  # 90 -> 0
                        else:
                            # Yellow to red gradient (0.5-1.0)
                            r = 255
                            g = int(255 * (1 - (norm_val - 0.5) * 2))  # 255 -> 0
                            b = 0
                        
                        # Add indicator for extreme values
                        if num_val < 15:  # Very low VIX (calm)
                            cell_style += "border-left: 3px solid #228B22;"  # Forest Green
                        elif num_val > 30:  # Very high VIX (fear)
                            cell_style += "border-left: 3px solid #8B0000;"  # Dark Red
                            
                        applied_color = True
                        
                    elif col_name == 'RSI':
                        # RSI normalization (0-100 scale)
                        # For RSI, we'll create a dual extreme color scheme:
                        # - RSI below 30 (oversold) should be deep red (extreme fear)
                        # - RSI above 70 (overbought) should be deep green (extreme greed)
                        # - RSI in middle range (40-60) should be neutral/yellow
                        
                        # Different color scheme for RSI:
                        # 0-30: Deep red to light red (oversold)
                        # 30-50: Light red to yellow (approaching neutral)
                        # 50-70: Yellow to light green (approaching overbought)
                        # 70-100: Light green to deep green (overbought)
                        
                        # First set the base colors based on the RSI range
                        if num_val <= 30:
                            # Oversold range (0-30) - deep red to light red
                            # Normalize to 0-1 range within this segment
                            segment_norm = 1.0 - (num_val / 30.0)
                            # Apply power function to increase contrast at lower values
                            intensity = segment_norm ** 0.7  # Increase color intensity at lower RSI
                            # Red gradient (deep red to lighter red)
                            r = 255
                            g = int(80 + (175 * (1 - intensity)))  # 80 (deep red) to 255 (light red)
                            b = int(80 * (1 - intensity))          # Keep some blue for color depth
                            
                        elif num_val <= 50:
                            # Approaching neutral from oversold (30-50) - light red to yellow
                            # Normalize to 0-1 range within this segment
                            segment_norm = (num_val - 30) / 20.0  # 0 at RSI=30, 1 at RSI=50
                            # Red to yellow gradient
                            r = 255
                            g = int(160 + (95 * segment_norm))  # 160 to 255
                            b = 0
                            
                        elif num_val <= 70:
                            # Approaching overbought (50-70) - yellow to light green
                            # Normalize to 0-1 range within this segment
                            segment_norm = (num_val - 50) / 20.0  # 0 at RSI=50, 1 at RSI=70
                            # Yellow to light green gradient
                            r = int(255 * (1 - segment_norm))    # 255 to 0
                            g = 255
                            b = 0
                            
                        else:
                            # Overbought range (70-100) - light green to deep green
                            # Normalize to 0-1 range within this segment
                            segment_norm = (num_val - 70) / 30.0  # 0 at RSI=70, 1 at RSI=100
                            # Apply power function to increase contrast at higher values
                            intensity = segment_norm ** 0.7  # Increase color intensity at higher RSI
                            # Green gradient (light green to deep green)
                            r = int(100 * (1 - intensity))    # Keep some red for color depth
                            g = int(255 - (155 * intensity))  # 255 (light green) to 100 (deep green)
                            b = int(50 * (1 - intensity))     # Keep some blue for color depth
                        
                        # Ensure valid RGB values
                        r = max(0, min(255, r))
                        g = max(0, min(255, g))
                        b = max(0, min(255, b))
                        
                        # Add indicator for extreme values
                        if num_val < 30:  # Oversold
                            cell_style += "border-left: 3px solid #8B0000;"  # Dark Red (oversold)
                        elif num_val > 70:  # Overbought
                            cell_style += "border-left: 3px solid #006400;"  # Dark Green (overbought)
                            
                        applied_color = True
                        
                    elif col_name == 'Volume':
                        # Volume normalization (billions scale)
                        # High volume indicates high market activity/potential stress
                        # We want highest volume to be deep red
                        
                        vol_in_billions = num_val / 1000000000
                        
                        # Use log scale to compress range, then normalize to 0-1
                        # Add 0.1 to avoid log(0) issues
                        log_vol = np.log10(vol_in_billions + 0.1)
                        
                        # Normalize to 0-1 range where 1 = highest volume
                        # Typical log(volume) range is -1 to 1.5
                        vol_norm = max(0.0, min(1.0, (log_vol + 1) / 2.5))
                        
                        # Apply power function to increase contrast 
                        norm_val = vol_norm ** 0.8  # Higher volume = higher norm_val = more red
                        
                        # Calculate colors based on normalized value
                        if norm_val < 0.5:
                            # Green to yellow gradient (0-0.5)
                            r = int(60 + 195 * norm_val * 2)  # 60 -> 255
                            g = int(200 + 55 * norm_val * 2)  # 200 -> 255
                            b = int(90 * (1 - norm_val * 2))  # 90 -> 0
                        else:
                            # Yellow to red gradient (0.5-1.0)
                            r = 255
                            g = int(255 * (1 - (norm_val - 0.5) * 2))  # 255 -> 0
                            b = 0
                        
                        # Add indicator for extreme values
                        if vol_in_billions < 0.5:  # Very low volume (< 500M)
                            cell_style += "border-left: 3px solid #228B22;"  # Forest Green
                        elif vol_in_billions > 5.0:  # Very high volume (> 5B)
                            cell_style += "border-left: 3px solid #8B0000;"  # Dark Red
                            
                        applied_color = True
                        
                    # Standard green-to-red gradient for percentage return columns    
                    elif '%' in col_name or 'Drop' in col_name:
                        if num_val < 0:
                            # Calculate intensity for negative values (red)
                            intensity = min(abs(num_val) / 10.0, 1.0)  # Max intensity at -10% or lower
                            # Create RGB components for a red background with varying intensity
                            r = 255
                            g = int(255 * (1 - intensity * 0.6))  # Keep some green to avoid pure red
                            b = int(255 * (1 - intensity * 0.8))  # Less blue for more red appearance
                        elif num_val > 0:
                            # Calculate intensity for positive values (green)
                            intensity = min(abs(num_val) / 10.0, 1.0)  # Max intensity at +10% or higher
                            # Create RGB components for a green background with varying intensity
                            r = int(255 * (1 - intensity * 0.8))  # Less red for more green appearance
                            g = 255
                            b = int(255 * (1 - intensity * 0.6))  # Keep some blue to avoid pure green
                            
                        applied_color = True
                    
                    # Apply the color if we processed this column
                    if applied_color:
                        cell_style += f"background-color:rgb({r},{g},{b}); "
                        
                        # Set text color based on background brightness for readability
                        brightness = (r * 299 + g * 587 + b * 114) / 1000  # Weighted luminance formula
                        text_color = "white" if brightness < 140 else "black"
                        cell_style += f"color:{text_color}; font-weight:500; "
                        
            except (ValueError, AttributeError):
                pass  # Not a number or empty, keep default styling
            
            # Add special formatting for the Total column (after applying colors)
            if is_total_col:
                cell_style += "border-left:2px solid #333; background-color:#f0f4f8; font-weight:900; color:#0056b3; text-shadow:0 0 0.2px #0056b3; "
            
            # Add special formatting for the totals row
            if is_last_row:
                cell_style += "border-top:2px solid #333; border-bottom:2px solid #333; background-color:#f0f4f8; font-weight:900; font-size:11px; color:#0056b3; text-shadow:0 0 0.2px #0056b3; "
            
            # Add extra emphasis for the intersection of totals row and column
            if is_last_row and is_total_col:
                cell_style = "font-size:11px; padding:2px 5px; white-space:nowrap; border-top:2px solid #333; border-bottom:2px solid #333; border-left:2px solid #333; background-color:#e6f0ff; font-weight:900; color:#004494; text-shadow:0 0 0.5px #004494; "
            
            html_content += f"<td style='{cell_style}'>{val}</td>"
        
        html_content += "</tr>"
    
    html_content += "</table></div>"
    
    # Display the custom HTML table
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Add download button for the detailed database with unique key
    if not events_df.empty:
        # Get the active market index for a unique key and proper file naming
        active_index = st.session_state.active_index if 'active_index' in st.session_state else 'sp500'
        market_name = {
            'sp500': 'S&P 500',
            'nasdaq': 'NASDAQ',
            'dow': 'Dow Jones'
        }.get(active_index, 'Market')
        
        csv = events_df.to_csv(index=False)
        st.download_button(
            label=f"Download {market_name} Drop Events Database",
            data=csv,
            file_name=f"{active_index}_drop_events_{st.session_state.drop_threshold}pct.csv",
            mime="text/csv",
            key=f"download_events_{active_index}"
        )
