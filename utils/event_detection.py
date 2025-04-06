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
            'severity': get_drop_severity(abs(row['Return']))
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
    Detect consecutive days of drops in the S&P 500
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with 'Close' and 'Return' columns
    threshold_pct : float
        Minimum percentage drop for each day to be considered
    num_days : int
        Number of consecutive days required
        
    Returns:
    --------
    list
        List of consecutive drop events with dates and magnitudes
    """
    import pandas as pd
    import numpy as np
    import streamlit as st
    
    # Print debug info
    print(f"Detecting consecutive drops with threshold: {threshold_pct}% and {num_days} days")
    
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
        
        # Calculate overall price drop from open of first day to close of last day
        start_price = window.iloc[0]['Open']
        end_price = window.iloc[-1]['Close']
        price_change_pct = (end_price / start_price - 1) * 100
        
        # For a window to be a valid consecutive drop:
        # 1. The total price change must be a drop at least as large as threshold * num_days * 0.6
        # 2. At least half of the days must have drops of at least threshold
        
        # Calculate minimum required drop for this window
        min_required_drop = -threshold_pct * num_days * 0.6
        
        # Count days with significant drops
        drop_days = sum(window['Return'] <= -threshold_pct)
        
        # Only consider this a valid consecutive drop if:
        # 1. The price dropped significantly overall
        # 2. At least half the days had significant drops
        # 3. The price change wasn't too extreme (to filter out data errors)
        if (price_change_pct <= min_required_drop and 
            drop_days >= num_days * 0.5 and 
            price_change_pct > -50):  # Sanity check
            
            # Create the event
            event = {
                'date': window.index[-1],  # End date
                'start_date': window.index[0],  # Start date
                'type': 'consecutive',
                'num_days': num_days,
                'daily_drops': window['Return'].tolist(),
                'cumulative_drop': price_change_pct,
                'close': window.iloc[-1]['Close'],
                'open': window.iloc[0]['Open'],
                'volume': window['Volume'].sum(),
                'severity': get_drop_severity(abs(price_change_pct))
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
    print(f"Found {len(filtered_events)} valid consecutive drop events for {num_days} days and {threshold_pct}% threshold")
    
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

def get_all_events():
    """
    Combine single-day and consecutive drop events
    
    Returns:
    --------
    list
        Combined list of all drop events
    """
    all_events = []
    
    # Add single-day drop events
    if st.session_state.drop_events:
        all_events.extend(st.session_state.drop_events)
    
    # Add consecutive drop events
    if st.session_state.consecutive_drop_events:
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
        return f"{date_str}: {event['severity']} Drop ({event['drop_pct']:.2f}%)"
    else:
        start_date = event['start_date'].strftime('%Y-%m-%d')
        return f"{start_date} to {date_str}: {event['severity']} Drop ({event['cumulative_drop']:.2f}% over {event['num_days']} days)"
