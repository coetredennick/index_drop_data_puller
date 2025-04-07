import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

def create_price_chart(data, drop_events=None, title="S&P 500 Price Chart", height=700):
    """
    Create an interactive price chart with marked drop events, VIX, and RSI panels
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data
    drop_events : list, optional
        List of drop events to mark on the chart
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive price chart
    """
    # Create figure with 3 rows for price+volume, VIX, and RSI
    fig = make_subplots(
        rows=3, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[
            [{"secondary_y": True}],  # Price chart with volume on secondary y-axis
            [{"secondary_y": False}],  # VIX
            [{"secondary_y": False}]   # RSI
        ],
        subplot_titles=["S&P 500 Price & Volume", "VIX (Volatility)", "RSI (14-day)"]
    )
    
    # PANEL 1: PRICE & VOLUME
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name="S&P 500",
            line=dict(color='#0E6EFD', width=1.5),
            opacity=0.8,
        ),
        row=1, col=1, secondary_y=False,
    )
    
    # Add volume trace on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker=dict(color='rgba(192, 192, 192, 0.5)'),
            opacity=0.4,
        ),
        row=1, col=1, secondary_y=True,
    )
    
    # PANEL 2: VIX (if available)
    if 'ATR_Pct' in data.columns:  # Use ATR as a proxy for volatility if VIX not available
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['ATR_Pct'],
                name="Volatility (ATR)",
                line=dict(color='#FF9800', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 152, 0, 0.1)',
            ),
            row=2, col=1, secondary_y=False,
        )
    
    # PANEL 3: RSI
    if 'RSI_14' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI_14'],
                name="RSI (14)",
                line=dict(color='#673AB7', width=1.5),
            ),
            row=3, col=1, secondary_y=False,
        )
        
        # Add horizontal lines for RSI at 30 and 70 (typical oversold/overbought thresholds)
        fig.add_shape(
            type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=3, col=1
        )
        fig.add_shape(
            type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=3, col=1
        )
    
    # Add drop event markers if provided
    if drop_events:
        # Check if there's a filter in session state
        current_filter = 'all'
        if 'current_event_type_filter' in st.session_state:
            current_filter = st.session_state.current_event_type_filter
        
        # Extract single-day drop events (only if allowed by filter)
        if current_filter in ['all', 'single_day']:
            single_day_events = [e for e in drop_events if e['type'] == 'single_day']
        else:
            single_day_events = []
            
        if single_day_events:
            event_dates = [e['date'] for e in single_day_events]
            event_prices = [data.loc[date, 'Close'] if date in data.index else None for date in event_dates]
            event_sizes = [abs(e['drop_pct'])*2 for e in single_day_events]  # Size based on drop magnitude
            event_colors = ['rgba(255, 0, 0, 0.8)' for _ in single_day_events]
            
            # Add markers to price chart
            fig.add_trace(
                go.Scatter(
                    x=event_dates,
                    y=event_prices,
                    mode='markers',
                    marker=dict(
                        size=event_sizes,
                        color=event_colors,
                        line=dict(width=1, color='DarkSlateGrey'),
                        symbol='triangle-down'
                    ),
                    name="Drop Events",
                    text=[f"{e['date'].strftime('%Y-%m-%d')}: {e['drop_pct']:.2f}%" for e in single_day_events],
                    hoverinfo="text",
                ),
                row=1, col=1, secondary_y=False,
            )
            
            # Also add markers to the VIX panel
            if 'ATR_Pct' in data.columns:
                vix_values = [data.loc[date, 'ATR_Pct'] if date in data.index and not pd.isna(data.loc[date, 'ATR_Pct']) else None for date in event_dates]
                
                # Filter out None values
                valid_dates = []
                valid_values = []
                valid_sizes = []
                valid_texts = []
                
                for i, (date, value, size, event) in enumerate(zip(event_dates, vix_values, event_sizes, single_day_events)):
                    if value is not None:
                        valid_dates.append(date)
                        valid_values.append(value)
                        valid_sizes.append(size)
                        valid_texts.append(f"{event['date'].strftime('%Y-%m-%d')}: {event['drop_pct']:.2f}%")
                
                if valid_dates:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_values,
                            mode='markers',
                            marker=dict(
                                size=[s*0.8 for s in valid_sizes],  # Slightly smaller
                                color='rgba(255, 0, 0, 0.8)',
                                symbol='triangle-down'
                            ),
                            name="",
                            text=valid_texts,
                            hoverinfo="text",
                            showlegend=False,
                        ),
                        row=2, col=1,
                    )
            
            # Also add markers to the RSI panel
            if 'RSI_14' in data.columns:
                rsi_values = [data.loc[date, 'RSI_14'] if date in data.index and not pd.isna(data.loc[date, 'RSI_14']) else None for date in event_dates]
                
                # Filter out None values
                valid_dates = []
                valid_values = []
                valid_sizes = []
                valid_texts = []
                
                for i, (date, value, size, event) in enumerate(zip(event_dates, rsi_values, event_sizes, single_day_events)):
                    if value is not None:
                        valid_dates.append(date)
                        valid_values.append(value)
                        valid_sizes.append(size)
                        valid_texts.append(f"{event['date'].strftime('%Y-%m-%d')}: {event['drop_pct']:.2f}%")
                
                if valid_dates:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_values,
                            mode='markers',
                            marker=dict(
                                size=[s*0.8 for s in valid_sizes],  # Slightly smaller
                                color='rgba(255, 0, 0, 0.8)',
                                symbol='triangle-down'
                            ),
                            name="",
                            text=valid_texts,
                            hoverinfo="text",
                            showlegend=False,
                        ),
                        row=3, col=1,
                    )
        
        # Extract consecutive drop events (only if allowed by filter)
        if current_filter in ['all', 'consecutive']:
            consecutive_events = [e for e in drop_events if e['type'] == 'consecutive']
        else:
            consecutive_events = []
            
        if consecutive_events:
            for event in consecutive_events:
                # Add rectangle to highlight consecutive drop period
                start_date = event['start_date']
                end_date = event['date']
                
                # Get y-range for the rectangle
                if start_date in data.index and end_date in data.index:
                    period_data = data.loc[start_date:end_date]
                    y_min = period_data['Low'].min() * 0.98
                    y_max = period_data['High'].max() * 1.02
                    
                    # Highlight on the price chart
                    fig.add_shape(
                        type="rect",
                        x0=start_date,
                        x1=end_date,
                        y0=y_min,
                        y1=y_max,
                        line=dict(
                            color="rgba(255, 0, 0, 0.3)",
                            width=2,
                        ),
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        opacity=0.5,
                        layer="below",
                        row=1, col=1
                    )
                    
                    # Highlight the same period on VIX panel
                    if 'ATR_Pct' in data.columns:
                        vix_period_data = data.loc[start_date:end_date]
                        vix_min = vix_period_data['ATR_Pct'].min() * 0.9 if not pd.isna(vix_period_data['ATR_Pct'].min()) else 0
                        vix_max = vix_period_data['ATR_Pct'].max() * 1.1 if not pd.isna(vix_period_data['ATR_Pct'].max()) else 0
                        
                        fig.add_shape(
                            type="rect",
                            x0=start_date,
                            x1=end_date,
                            y0=vix_min,
                            y1=vix_max,
                            line=dict(
                                color="rgba(255, 0, 0, 0.3)",
                                width=1,
                            ),
                            fillcolor="rgba(255, 0, 0, 0.1)",
                            opacity=0.5,
                            layer="below",
                            row=2, col=1
                        )
                    
                    # Highlight the same period on RSI panel
                    if 'RSI_14' in data.columns:
                        rsi_period_data = data.loc[start_date:end_date]
                        rsi_min = rsi_period_data['RSI_14'].min() * 0.9 if not pd.isna(rsi_period_data['RSI_14'].min()) else 0
                        rsi_max = rsi_period_data['RSI_14'].max() * 1.1 if not pd.isna(rsi_period_data['RSI_14'].max()) else 100
                        
                        fig.add_shape(
                            type="rect",
                            x0=start_date,
                            x1=end_date,
                            y0=rsi_min,
                            y1=rsi_max,
                            line=dict(
                                color="rgba(255, 0, 0, 0.3)",
                                width=1,
                            ),
                            fillcolor="rgba(255, 0, 0, 0.1)",
                            opacity=0.5,
                            layer="below",
                            row=3, col=1
                        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        # Update subplot y-axis titles
        yaxis=dict(title="Price ($)"),
        yaxis2=dict(title="Volume", showgrid=False, tickformat=".2s"),
        yaxis3=dict(title="Volatility (%)"),
        yaxis4=dict(title="RSI"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Configure y-axes
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(
        title_text="Volume", 
        secondary_y=True, 
        showgrid=False,
        tickformat=".2s"
    )
    
    return fig

def create_returns_heatmap(returns_data, title="Post-Drop Returns (%)"):
    """
    Create a heatmap visualization of returns after drop events
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing return data for different time periods
    title : str, optional
        Chart title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Heatmap visualization
    """
    # Create figure
    fig = go.Figure()
    
    # Define time periods in order
    time_periods = ['1W', '1M', '3M', '6M', '1Y', '3Y']
    
    # Create heatmap cells
    for i, period in enumerate(time_periods):
        for j, (date, row) in enumerate(returns_data.iterrows()):
            value = row[f'fwd_return_{period.lower()}'] if f'fwd_return_{period.lower()}' in row else None
            
            if value is not None:
                # Determine cell color based on return value
                if value > 0:
                    color = f'rgba(0, 128, 0, {min(abs(value) / 20, 1)})'  # Green for positive returns
                else:
                    color = f'rgba(255, 0, 0, {min(abs(value) / 20, 1)})'  # Red for negative returns
                
                # Add cell
                fig.add_trace(
                    go.Scatter(
                        x=[j],
                        y=[i],
                        mode='markers',
                        marker=dict(
                            size=30,
                            color=color,
                            symbol='square',
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        text=f"{value:.2f}%",
                        hoverinfo="text",
                        showlegend=False
                    )
                )
                
                # Add text annotation
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{value:.1f}%",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=10,
                        color="white" if abs(value) > 5 else "black"
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=800,
        height=400,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(returns_data))),
            ticktext=[d.strftime('%Y-%m-%d') for d in returns_data.index],
            tickangle=45
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(time_periods))),
            ticktext=time_periods
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=50, b=100),
    )
    
    return fig

def create_recovery_chart(data, event, title="Post-Drop Recovery", height=400):
    """
    Create a chart showing the recovery trajectory after a drop event
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data
    event : dict
        Drop event data
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Recovery chart
    """
    # Error handling: Check if event is None or empty
    if event is None or not isinstance(event, dict):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No event data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Determine the range of data to display
    try:
        if event['type'] == 'single_day':
            # Ensure date is a pandas Timestamp
            event_date = pd.Timestamp(event['date']) if not isinstance(event['date'], pd.Timestamp) else event['date']
            start_date = event_date - pd.Timedelta(days=5)
        else:
            # Ensure dates are pandas Timestamps
            event_date = pd.Timestamp(event['date']) if not isinstance(event['date'], pd.Timestamp) else event['date']
            start_date = pd.Timestamp(event['start_date']) if not isinstance(event['start_date'], pd.Timestamp) else event['start_date']
            start_date = start_date - pd.Timedelta(days=5)
        
        # Get one year of data after the event or as much as is available
        end_date = event_date + pd.Timedelta(days=365)
        
        # Ensure we don't try to get data from the future
        now_ts = pd.Timestamp(datetime.now())
        if end_date > now_ts:
            end_date = now_ts
        
        # Ensure the data index is using pandas Timestamps for consistent comparison
        data_copy = data.copy()
        data_copy.index = pd.DatetimeIndex([pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx for idx in data.index])
        
        # Ensure start_date and end_date are pandas Timestamps for comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Filter data for the selected period using consistent timestamp format
        mask = (data_copy.index >= start_ts) & (data_copy.index <= end_ts)
        period_data = data_copy.loc[mask].copy()
    except Exception as e:
        # Handle any errors in date processing
        print(f"Error processing dates in recovery chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text=f"Error processing event dates: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    if period_data.empty:
        # No data available
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No data available for this period",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Normalize prices to the event date for percentage change visualization
    if event_date in period_data.index:
        event_price = period_data.loc[event_date, 'Close']
        period_data['Normalized'] = (period_data['Close'] / event_price - 1) * 100
    else:
        # If event date is not in the data (weekend, holiday), use the closest date
        closest_date = period_data.index[period_data.index.get_indexer([event_date], method='nearest')[0]]
        event_price = period_data.loc[closest_date, 'Close']
        period_data['Normalized'] = (period_data['Close'] / event_price - 1) * 100
        event_date = closest_date  # Update event_date to the closest date in the data
    
    # Create figure
    fig = go.Figure()
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=period_data.index,
            y=period_data['Normalized'],
            name="S&P 500",
            line=dict(color='#0E6EFD', width=1.5),
            opacity=0.8,
        )
    )
    
    # Add a reference line at 0 (event day level)
    fig.add_shape(
        type="line",
        x0=period_data.index[0],
        x1=period_data.index[-1],
        y0=0,
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        ),
    )
    
    # Mark the event date(s)
    if event['type'] == 'single_day':
        # Single day event
        fig.add_trace(
            go.Scatter(
                x=[event_date],
                y=[0],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='triangle-down'
                ),
                name=f"Drop Event ({event['drop_pct']:.2f}%)",
                hoverinfo="name",
            )
        )
    else:
        # Consecutive drop event - mark start and end dates
        try:
            # Ensure date is a pandas Timestamp
            start_date = pd.Timestamp(event['start_date']) if not isinstance(event['start_date'], pd.Timestamp) else event['start_date']
            
            # Removed this section because it's now handled in the markers section
            
            # Add area to highlight the drop period
            # Make sure we're dealing with valid date range within our data
            range_start = max(start_date, period_data.index[0])
            range_end = min(event_date, period_data.index[-1])
            
            # Only proceed if we have a valid date range
            if range_start <= range_end and not period_data.loc[range_start:range_end].empty:
                y_min = min(0, period_data.loc[range_start:range_end, 'Normalized'].min()) - 1
                y_max = max(0, period_data.loc[range_start:range_end, 'Normalized'].max()) + 1
                
                fig.add_shape(
                    type="rect",
                    x0=start_date,
                    x1=event_date,
                    y0=y_min,
                    y1=y_max,
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(color="rgba(255, 0, 0, 0.5)"),
                    layer="below",
                )
        except Exception as e:
            print(f"Error highlighting consecutive drop period: {e}")
            # Continue without the highlighting if there's an error
        
        # Add markers for start and end dates with a safety check for start_normalized
        try:
            # Initialize start_normalized with a default value of None
            start_normalized = None
            
            # Calculate start_normalized (safely moved to this section)
            if start_date in period_data.index and 'Close' in period_data.columns:
                start_price = period_data.loc[start_date, 'Close']
                if pd.notna(start_price) and pd.notna(event_price) and event_price != 0:  # Avoid division by zero
                    start_normalized = (start_price / event_price - 1) * 100
                    
            # Ensure start_normalized is defined and has a value
            if start_normalized is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[start_date, event_date],
                        y=[start_normalized, 0],
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=['orange', 'red'],
                            symbol=['circle', 'triangle-down']
                        ),
                        text=['Start', 'End'],
                        textposition='top center',
                        name=f"Drop Period ({event['cumulative_drop']:.2f}% over {event['num_days']} days)",
                        hoverinfo="name",
                    )
                )
            else:
                print("Warning: start_normalized is None, cannot add start/end markers")
        except Exception as e:
            print(f"Error adding consecutive drop markers in recovery chart: {e}")
    
    # Add reference lines for standard time periods
    time_periods = [
        (30, '1M', 'rgba(0,0,0,0.2)'),
        (90, '3M', 'rgba(0,0,0,0.3)'),
        (180, '6M', 'rgba(0,0,0,0.4)'),
        (365, '1Y', 'rgba(0,0,0,0.5)')
    ]
    
    for days, label, color in time_periods:
        try:
            # Calculate the date for this time period using pandas Timedelta for consistency
            period_date = event_date + pd.Timedelta(days=days)
            
            # Only add if the date is within our data range
            if period_date <= period_data.index[-1]:
                y_min = min(period_data['Normalized']) - 2
                y_max = max(period_data['Normalized']) + 2
                
                fig.add_shape(
                    type="line",
                    x0=period_date,
                    x1=period_date,
                    y0=y_min,
                    y1=y_max,
                    line=dict(
                        color=color,
                        width=1,
                        dash="dot",
                    ),
                )
                
                # Add text annotation
                fig.add_annotation(
                    x=period_date,
                    y=y_max,
                    text=label,
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10, color="black")
                )
        except Exception as e:
            print(f"Error adding reference line for {label}: {e}")
            # Continue without this reference line if there's an error
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="% Change from Drop Day",
        hovermode="x unified",
        height=height,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Add custom hover template to display the exact % change
    fig.update_traces(
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>",
    )
    
    return fig

def create_technical_indicator_chart(data, event, indicator, title=None, height=300):
    """
    Create a chart showing the behavior of a technical indicator around a drop event
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing S&P 500 historical data with technical indicators
    event : dict
        Drop event data
    indicator : str
        Name of the technical indicator to display
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Technical indicator chart
    """
    # Default title if none provided
    if title is None:
        title = f"{indicator} Around Drop Event"
    
    # Error handling: Check if event is None or empty
    if event is None or not isinstance(event, dict):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No event data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Determine the range of data to display
    try:
        if event['type'] == 'single_day':
            # Ensure date is a pandas Timestamp
            event_date = pd.Timestamp(event['date']) if not isinstance(event['date'], pd.Timestamp) else event['date']
            start_date = event_date - pd.Timedelta(days=30)
        else:
            # Ensure dates are pandas Timestamps
            event_date = pd.Timestamp(event['date']) if not isinstance(event['date'], pd.Timestamp) else event['date']
            start_date = pd.Timestamp(event['start_date']) if not isinstance(event['start_date'], pd.Timestamp) else event['start_date']
            start_date = start_date - pd.Timedelta(days=30)
        
        # Get 60 days of data after the event or as much as is available
        end_date = event_date + pd.Timedelta(days=60)
        
        # Ensure we don't try to get data from the future
        if end_date > pd.Timestamp(datetime.now()):
            end_date = pd.Timestamp(datetime.now())
        
        # Ensure the data index is using pandas Timestamps for consistent comparison
        data_copy = data.copy()
        data_copy.index = pd.DatetimeIndex([pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx for idx in data.index])
        
        # Ensure start_date and end_date are pandas Timestamps for comparison
        start_ts = pd.Timestamp(start_date) if not isinstance(start_date, pd.Timestamp) else start_date
        end_ts = pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date
        
        # Filter data for the selected period
        mask = (data_copy.index >= start_ts) & (data_copy.index <= end_ts)
        period_data = data_copy.loc[mask].copy()
    except Exception as e:
        # Handle any errors in date processing
        print(f"Error processing dates in technical indicator chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text=f"Error processing event dates: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    if period_data.empty or indicator not in period_data.columns:
        # No data available
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text=f"No {indicator} data available for this period",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Create figure with secondary y-axis for price
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add indicator trace
    fig.add_trace(
        go.Scatter(
            x=period_data.index,
            y=period_data[indicator],
            name=indicator,
            line=dict(color='purple', width=1.5),
        ),
        secondary_y=False,
    )
    
    # Add price trace on secondary axis
    fig.add_trace(
        go.Scatter(
            x=period_data.index,
            y=period_data['Close'],
            name="S&P 500",
            line=dict(color='#0E6EFD', width=1),
            opacity=0.6,
        ),
        secondary_y=True,
    )
    
    # Mark the event date(s)
    if event['type'] == 'single_day':
        # Single day event
        fig.add_trace(
            go.Scatter(
                x=[event_date],
                y=[period_data.loc[event_date, indicator] if event_date in period_data.index else None],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='triangle-down'
                ),
                name=f"Drop Event ({event['drop_pct']:.2f}%)",
                hoverinfo="name",
            ),
            secondary_y=False,
        )
    else:
        # Consecutive drop event - mark start and end dates
        try:
            # Ensure date is a pandas Timestamp
            start_date = pd.Timestamp(event['start_date']) if not isinstance(event['start_date'], pd.Timestamp) else event['start_date']
            
            # Add markers for start and end dates
            marker_y = [None, None]
            
            # Safely get values for the start date
            if start_date in period_data.index and indicator in period_data.columns:
                marker_y[0] = period_data.loc[start_date, indicator]
            
            # Safely get values for the event date
            if event_date in period_data.index and indicator in period_data.columns:
                marker_y[1] = period_data.loc[event_date, indicator]
                
            fig.add_trace(
                go.Scatter(
                    x=[start_date, event_date],
                    y=marker_y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=['orange', 'red'],
                        symbol=['circle', 'triangle-down']
                    ),
                    name=f"Drop Period ({event['cumulative_drop']:.2f}% over {event['num_days']} days)",
                    hoverinfo="name",
                ),
                secondary_y=False,
            )
        except Exception as e:
            print(f"Error adding consecutive drop markers: {e}")
            # Continue without these markers if there's an error
    
    # Add reference lines for indicator if needed
    if indicator == 'RSI_14':
        # Add overbought/oversold lines for RSI
        for level, color in [(30, 'green'), (70, 'red')]:
            fig.add_shape(
                type="line",
                x0=period_data.index[0],
                x1=period_data.index[-1],
                y0=level,
                y1=level,
                line=dict(
                    color=color,
                    width=1,
                    dash="dash",
                ),
            )
    elif indicator == 'STOCHk_14_3_3':
        # Add overbought/oversold lines for Stochastic
        for level, color in [(20, 'green'), (80, 'red')]:
            fig.add_shape(
                type="line",
                x0=period_data.index[0],
                x1=period_data.index[-1],
                y0=level,
                y1=level,
                line=dict(
                    color=color,
                    width=1,
                    dash="dash",
                ),
            )
    elif indicator == 'MACDh_12_26_9':
        # Add zero line for MACD Histogram
        fig.add_shape(
            type="line",
            x0=period_data.index[0],
            x1=period_data.index[-1],
            y0=0,
            y1=0,
            line=dict(
                color="gray",
                width=1,
                dash="dash",
            ),
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Configure y-axes
    fig.update_yaxes(title_text=indicator, secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
    
    return fig

def create_distribution_histogram(drop_events, title="Distribution of Drop Events", height=400):
    """
    Create a histogram showing the distribution of drop events by magnitude
    
    Parameters:
    -----------
    drop_events : list
        List of drop events
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Histogram visualization
    """
    if not drop_events:
        # No data available
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No drop events to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Check if there's a filter in session state
    current_filter = 'all'
    if 'current_event_type_filter' in st.session_state:
        current_filter = st.session_state.current_event_type_filter
    
    # Filter events based on the current filter
    filtered_events = []
    if current_filter == 'all':
        filtered_events = drop_events
    else:
        filtered_events = [e for e in drop_events if e['type'] == current_filter]
    
    # If no events match the filter
    if not filtered_events:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text=f"No {current_filter} events to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=height
        )
        return fig
    
    # Extract drop percentages
    drops = []
    for event in filtered_events:
        if event['type'] == 'single_day':
            drops.append(abs(event['drop_pct']))
        else:
            drops.append(abs(event['cumulative_drop']))
    
    # Create histogram bins
    bins = np.arange(0, max(drops) + 1, 0.5)
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(
        go.Histogram(
            x=drops,
            name="Drop Events",
            marker_color='rgba(255, 0, 0, 0.7)',
            xbins=dict(
                start=min(bins),
                end=max(bins),
                size=0.5
            ),
            opacity=0.7
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Drop Magnitude (%)",
        yaxis_title="Frequency",
        height=height,
        template="plotly_white",
        bargap=0.1,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Add vertical lines for severity categories
    severity_levels = [(3, 'Significant'), (5, 'Major'), (7, 'Severe')]
    for level, label in severity_levels:
        fig.add_shape(
            type="line",
            x0=level,
            x1=level,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="rgba(0, 0, 0, 0.5)",
                width=1,
                dash="dash",
            ),
        )
        
        # Add text annotation
        fig.add_annotation(
            x=level,
            y=1,
            text=label,
            showarrow=False,
            yshift=10,
            xshift=10,
            textangle=-90,
            font=dict(size=10, color="black")
        )
    
    return fig
