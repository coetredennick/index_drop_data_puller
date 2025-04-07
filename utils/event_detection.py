import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

def detect_drop_events(data, threshold_pct):
    """
    Detect single-day drop events in the S&P 500
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with 'Close' and 'Return' columns
    threshold_pct : float
        Minimum percentage drop to be considered an event
        
    Returns:
    --------
    list
        List of drop events with dates and magnitudes
    """
    # Identify days with negative returns exceeding the threshold
    drop_days = data[data['Return'] <= -threshold_pct].copy()
    
    if drop_days.empty:
        return []
    
    # Create list of drop events
    events = []
    for date, row in drop_days.iterrows():
        # Get data for this event
        event = {
            'date': date,
            'type': 'single_day',
            'drop_pct': row['Return'],
            'close': row['Close'],
            'volume': row['Volume'],
            'severity': get_drop_severity(abs(row['Return'])),
            # Add rate of decline metrics (for single day, it's the same as the drop percentage)
            'decline_rate_per_day': abs(row['Return']),  # In % per day
            'decline_duration': 1  # 1 day by definition
        }
        
        # Add forward returns
        for period in ['1W', '1M', '3M', '6M', '1Y', '3Y']:
            column = f'Fwd_Ret_{period}'
            if column in row and not pd.isna(row[column]):
                event[f'fwd_return_{period.lower()}'] = row[column]
        
        # Add technical indicators
        for indicator in ['RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Volume_Ratio']:
            if indicator in row and not pd.isna(row[indicator]):
                event[indicator] = row[indicator]
        
        events.append(event)
    
    # Sort events by date (newest first)
    events.sort(key=lambda x: x['date'], reverse=True)
    
    return events

def detect_consecutive_drops(data, threshold_pct, num_days):
    """
    Detect consecutive days of drops in the S&P 500 where EACH day meets or exceeds the threshold.
    
    This algorithm focuses ONLY on the daily drop threshold. For example, with a 4.8% threshold
    and 2 consecutive days, it will find all events where there were at least 2 consecutive days
    each with a 4.8% or greater drop. The total cumulative drop is calculated but not used as a 
    filtering criterion.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with 'Close' and 'Return' columns
    threshold_pct : float
        Minimum percentage drop for EACH day to be considered (positive number)
    num_days : int
        Number of consecutive days required
        
    Returns:
    --------
    list
        List of consecutive drop events with dates and magnitudes where every day drops by at least threshold_pct
    """
    # Print debug info
    print(f"Detecting consecutive drops with threshold: {threshold_pct}% for EACH day over {num_days} consecutive days")
    
    if num_days < 2:
        return []
    
    # Rather than looking at individual drops, look at price movement in rolling windows
    # This approach more accurately captures market corrections
    
    # Initialize events list
    events = []
    
    # Get a sorted copy of the data
    sorted_data = data.sort_index()
    
    # Create a rolling window of size num_days
    for i in range(len(sorted_data) - num_days + 1):
        # Get the window
        window = sorted_data.iloc[i:i+num_days]
        
        # Skip if we don't have enough data
        if len(window) < num_days:
            continue
        
        # Calculate both methods of drop:
        # 1. Open-to-close: From open of first day to close of last day
        start_price = window.iloc[0]['Open']
        end_price = window.iloc[-1]['Close']
        price_change_pct = (end_price / start_price - 1) * 100
        
        # 2. Cumulative: Sum of the daily percentage returns
        # This better represents the actual drop experienced by the market
        cumulative_return_pct = window['Return'].sum()
        
        # NEW LOGIC: For a window to be a valid consecutive drop:
        # 1. EVERY day must have a drop at least as large as the threshold
        # Note: We're not requiring the cumulative drop to be a specific amount
        
        # Check if ALL days meet the threshold requirement
        all_days_meet_threshold = all(day_return <= -threshold_pct for day_return in window['Return'])
        
        # Only consider this a valid consecutive drop if:
        # 1. ALL days have drops meeting or exceeding the threshold
        # 2. The price change isn't too extreme (to filter out data errors)
        if (all_days_meet_threshold and
            price_change_pct > -50):  # Sanity check
            
            # Create the event
            event = {
                'date': window.index[-1],  # End date
                'start_date': window.index[0],  # Start date
                'type': 'consecutive',
                'num_days': num_days,
                'daily_drops': window['Return'].tolist(),
                'cumulative_drop': cumulative_return_pct,  # Using sum of daily returns
                'price_change_pct': price_change_pct,  # Keep the open-to-close change for reference
                'close': window.iloc[-1]['Close'],
                'open': window.iloc[0]['Open'],
                'volume': window['Volume'].sum(),
                'severity': get_drop_severity(abs(cumulative_return_pct)),  # Use cumulative return for severity
                # Add rate of decline metrics
                'decline_duration': num_days,  # Number of trading days of the decline
                'decline_rate_per_day': abs(cumulative_return_pct) / num_days,  # Average % decline per day
                'max_daily_decline': abs(min(window['Return'])),  # Largest single-day drop during the event
                'decline_acceleration': abs(window['Return'].iloc[-1]) - abs(window['Return'].iloc[0])  # Positive means accelerating drop
            }
            
            # Add forward returns from the last day of the drop
            last_row = window.iloc[-1]
            for period in ['1W', '1M', '3M', '6M', '1Y', '3Y']:
                column = f'Fwd_Ret_{period}'
                if column in last_row and not pd.isna(last_row[column]):
                    event[f'fwd_return_{period.lower()}'] = last_row[column]
            
            # Add technical indicators from the last day of the drop
            for indicator in ['RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Volume_Ratio']:
                if indicator in last_row and not pd.isna(last_row[indicator]):
                    event[indicator] = last_row[indicator]
            
            events.append(event)
    
    # For consecutive drops, we need to filter out overlapping periods
    # Sort by severity (largest drop first)
    events.sort(key=lambda x: abs(x['cumulative_drop']), reverse=True)
    
    # Filter out overlapping periods (keep the more severe ones)
    filtered_events = []
    used_dates = set()
    
    for event in events:
        # Create a set of all dates in this event
        event_dates = set(pd.date_range(start=event['start_date'], end=event['date']))
        
        # Check if this event overlaps with any already used dates
        if not event_dates.intersection(used_dates):
            filtered_events.append(event)
            used_dates.update(event_dates)
    
    # Sort by date (newest first)
    filtered_events.sort(key=lambda x: x['date'], reverse=True)
    
    # Print debug info
    print(f"Found {len(filtered_events)} valid consecutive drop events where EACH of {num_days} days dropped by at least {threshold_pct}%")
    
    return filtered_events

def get_drop_severity(drop_pct):
    """
    Categorize the severity of a market drop
    
    Parameters:
    -----------
    drop_pct : float
        Percentage drop (positive number)
        
    Returns:
    --------
    str
        Severity category ('Severe', 'Major', 'Significant', or 'Minor')
    """
    if drop_pct >= 7:
        return 'Severe'
    elif drop_pct >= 5:
        return 'Major'
    elif drop_pct >= 3:
        return 'Significant'
    else:
        return 'Minor'

def get_all_events(event_type='all'):
    """
    Combine single-day and consecutive drop events
    
    Parameters:
    -----------
    event_type : str, optional
        Type of events to include: 'all', 'single_day', or 'consecutive'
        
    Returns:
    --------
    list
        Filtered and sorted list of drop events
    """
    all_events = []
    
    # Add single-day drop events if requested
    if (event_type == 'all' or event_type == 'single_day') and st.session_state.drop_events:
        all_events.extend(st.session_state.drop_events)
    
    # Add consecutive drop events if requested
    if (event_type == 'all' or event_type == 'consecutive') and st.session_state.consecutive_drop_events:
        all_events.extend(st.session_state.consecutive_drop_events)
    
    # Sort by date (newest first)
    all_events.sort(key=lambda x: x['date'], reverse=True)
    
    return all_events

def get_event_label(event):
    """
    Create a readable label for a drop event
    
    Parameters:
    -----------
    event : dict
        Drop event data
        
    Returns:
    --------
    str
        Readable event label
    """
    date_str = event['date'].strftime('%Y-%m-%d')
    
    if event['type'] == 'single_day':
        # Include rate of decline in single-day event
        return f"{date_str}: {event['severity']} Drop ({event['drop_pct']:.2f}%)"
    else:
        start_date = event['start_date'].strftime('%Y-%m-%d')
        # Show the threshold that each day had to meet
        actual_threshold = st.session_state.drop_threshold if 'drop_threshold' in st.session_state else None
        
        # Add rate of decline information
        rate_info = f" | Rate: {event.get('decline_rate_per_day', 0):.2f}%/day"
        
        if actual_threshold is not None:
            return f"{start_date} to {date_str}: {event['severity']} Drop ({event['cumulative_drop']:.2f}% over {event['num_days']} days{rate_info}, EACH day ≥{actual_threshold:.1f}%)"
        else:
            # Fallback if threshold isn't in session state
            return f"{start_date} to {date_str}: {event['severity']} Drop ({event['cumulative_drop']:.2f}% over {event['num_days']} days{rate_info})"
