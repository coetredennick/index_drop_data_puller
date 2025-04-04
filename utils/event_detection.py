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
    if num_days < 2:
        return []
    
    # Identify consecutive days with negative returns
    events = []
    for i in range(len(data) - num_days + 1):
        window = data.iloc[i:i+num_days]
        
        # Check if all days in the window have drops exceeding the threshold
        if all(window['Return'] <= -threshold_pct):
            # Calculate cumulative drop
            cumulative_drop = (window.iloc[-1]['Close'] / window.iloc[0]['Open'] - 1) * 100
            
            # Create event
            event = {
                'date': window.index[-1],  # End date
                'start_date': window.index[0],  # Start date
                'type': 'consecutive',
                'num_days': num_days,
                'daily_drops': window['Return'].tolist(),
                'cumulative_drop': cumulative_drop,
                'close': window.iloc[-1]['Close'],
                'severity': get_drop_severity(abs(cumulative_drop))
            }
            
            # Add forward returns from the last day of the drop
            last_row = data.loc[window.index[-1]]
            for period in ['1W', '1M', '3M', '6M', '1Y', '3Y']:
                column = f'Fwd_Ret_{period}'
                if column in last_row and not pd.isna(last_row[column]):
                    event[f'fwd_return_{period.lower()}'] = last_row[column]
            
            # Add technical indicators from the last day
            for indicator in ['RSI_14', 'STOCHk_14_3_3', 'BBP_20_2', 'MACDh_12_26_9', 'ATR_Pct', 'Volume_Ratio']:
                if indicator in last_row and not pd.isna(last_row[indicator]):
                    event[indicator] = last_row[indicator]
            
            events.append(event)
    
    # Sort events by date (newest first)
    events.sort(key=lambda x: x['date'], reverse=True)
    
    return events

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
