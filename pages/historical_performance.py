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
            key="historical_event_type_filter"
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
                
                if pd.notna(num_val) and not is_last_row:  # Skip coloring for the totals row
                    if not is_total_col:  # Apply heatmap to all non-total columns
                        # Different color schemes based on column type
                        if col_name == 'VIX':
                            # Smooth gradient for VIX from green (low volatility) to yellow to red (high volatility)
                            # VIX ranges typically from 10 (very calm) to 80+ (extreme volatility)
                            # Use a normalized scale where 10=0.0, 50=1.0
                            
                            # Calculate normalized position in the range 0.0-1.0
                            norm_val = max(0.0, min(1.0, (num_val - 10) / 40.0))
                            
                            # Create gradient colors - going from blue-green (calm) to yellow to red (volatile)
                            if norm_val < 0.5:  # First half of gradient: blue-green to yellow
                                # Blue-green (0) to yellow (0.5)
                                ratio = norm_val * 2  # Scale to 0-1 range for this half
                                r = int(100 + 155 * ratio)  # 100 -> 255
                                g = int(200 + 55 * ratio)   # 200 -> 255
                                b = int(220 * (1 - ratio))  # 220 -> 0
                            else:  # Second half: yellow to red
                                # Yellow (0.5) to red (1.0)
                                ratio = (norm_val - 0.5) * 2  # Scale to 0-1 range for this half
                                r = 255
                                g = int(255 * (1 - ratio))  # 255 -> 0
                                b = 0
                            
                            cell_style += f"background-color:rgb({r},{g},{b}); "
                            
                            # Text color (black on lighter backgrounds, white on darker ones)
                            text_color = "white" if (norm_val > 0.7) else "black"
                            cell_style += f"color:{text_color}; font-weight:500; "
                            
                            # Add a small indicator of the value range
                            if num_val < 15:
                                cell_style += "border-left: 3px solid #88d8b0;" # Very low VIX indicator
                            elif num_val > 30:
                                cell_style += "border-left: 3px solid #ff5252;" # High VIX indicator
                        
                        elif col_name == 'RSI':
                            # Smooth gradient for RSI across the full 0-100 range
                            # 0 = extremely oversold (deep blue)
                            # 50 = neutral (white/light gray)
                            # 100 = extremely overbought (deep red)
                            
                            # Calculate position in gradient
                            norm_val = num_val / 100.0  # RSI is 0-100 scale
                            
                            # Create gradient colors
                            if norm_val < 0.3:  # Oversold zone (0-30)
                                # Deep blue to lighter blue
                                intensity = 1.0 - (norm_val / 0.3)  # 1.0 at RSI=0, 0.0 at RSI=30
                                r = int(65 + (150 * (1 - intensity)))
                                g = int(105 + (100 * (1 - intensity)))
                                b = int(225 - (50 * (1 - intensity)))
                                
                                # Add a blue border indicator for oversold
                                cell_style += "border-left: 3px solid #3a86ff;"
                                
                            elif norm_val > 0.7:  # Overbought zone (70-100)
                                # Light red to deep red
                                intensity = (norm_val - 0.7) / 0.3  # 0.0 at RSI=70, 1.0 at RSI=100
                                r = int(225 + (30 * intensity))
                                g = int(120 - (80 * intensity))
                                b = int(120 - (80 * intensity))
                                
                                # Add a red border indicator for overbought
                                cell_style += "border-left: 3px solid #ff3333;"
                                
                            else:  # Neutral zone (30-70)
                                # Balanced gradient from blue-white to white-red
                                # Scale to 0-1 within 30-70 range
                                scaled = (norm_val - 0.3) / 0.4
                                
                                if scaled < 0.5:  # 30-50 range (blue-tinted to white)
                                    subrange = scaled * 2  # 0-1 in 30-50 range
                                    r = int(175 + (80 * subrange))
                                    g = int(190 + (65 * subrange))
                                    b = int(230 + (25 * subrange))
                                else:  # 50-70 range (white to red-tinted)
                                    subrange = (scaled - 0.5) * 2  # 0-1 in 50-70 range
                                    r = int(255)
                                    g = int(255 - (135 * subrange))
                                    b = int(255 - (135 * subrange))
                            
                            # Text color
                            text_color = "white" if (norm_val < 0.2 or norm_val > 0.8) else "black"
                            cell_style += f"background-color:rgb({r},{g},{b}); "
                            cell_style += f"color:{text_color}; font-weight:500; "
                        
                        elif col_name == 'Volume':
                            # Calculate a normalized volume range (typical S&P 500 volume range)
                            # Assume normal volume range is between 1-10 billion for modern S&P
                            vol_in_billions = num_val / 1000000000
                            norm_vol = max(0.0, min(1.0, vol_in_billions / 5.0))  # Normalize 0-5B to 0-1 range
                            
                            # Create a smooth blue-purple gradient based on volume
                            # Low volume (light blue) to high volume (deep indigo-purple)
                            r = int(140 + (30 * (1 - norm_vol)))
                            g = int(160 + (30 * (1 - norm_vol)))
                            b = int(220 + (35 * norm_vol))
                            
                            cell_style += f"background-color:rgb({r},{g},{b}); "
                            
                            # Add a subtle border for exceptionally high or low volume
                            if norm_vol > 0.8:
                                cell_style += "border-left: 3px solid #6a4c93;" # High volume indicator
                            elif norm_vol < 0.2:
                                cell_style += "border-left: 3px solid #8ecae6;" # Low volume indicator
                            
                            # Text color - black works well for this color range
                            text_color = "black"
                            if norm_vol > 0.8:
                                text_color = "white"  # White text on very deep backgrounds
                                
                            cell_style += f"color:{text_color}; font-weight:500; "
                        
                        # Standard green-to-red heatmap for percentage returns
                        elif '%' in col_name or 'Drop' in col_name:
                            if num_val < 0:
                                # Calculate intensity for negative values (red)
                                intensity = min(abs(num_val) / 10.0, 1.0)  # Max intensity at -10% or lower
                                # Create RGB components for a red background with varying intensity
                                r = 255
                                g = int(255 * (1 - intensity * 0.6))  # Keep some green to avoid pure red
                                b = int(255 * (1 - intensity * 0.8))  # Less blue for more red appearance
                                cell_style += f"background-color:rgb({r},{g},{b}); "
                                # Add contrasting text color for better readability
                                cell_style += "color:#000000; font-weight:500; "  # Black text
                                
                            elif num_val > 0:
                                # Calculate intensity for positive values (green)
                                intensity = min(abs(num_val) / 10.0, 1.0)  # Max intensity at +10% or higher
                                # Create RGB components for a green background with varying intensity
                                r = int(255 * (1 - intensity * 0.8))  # Less red for more green appearance
                                g = 255
                                b = int(255 * (1 - intensity * 0.6))  # Keep some blue to avoid pure green
                                cell_style += f"background-color:rgb({r},{g},{b}); "
                                # Add contrasting text color for better readability
                                cell_style += "color:#000000; font-weight:500; "  # Black text
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
    
    # Add download button for the detailed database
    if not events_df.empty:
        csv = events_df.to_csv(index=False)
        st.download_button(
            label="Download Drop Events Database",
            data=csv,
            file_name=f"sp500_drop_events_{st.session_state.drop_threshold}pct.csv",
            mime="text/csv",
        )
