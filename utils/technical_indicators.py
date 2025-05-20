import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for the given data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing price and volume data for S&P 500
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicators
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Add strategy (all default values)
    strat = ta.Strategy(
        name="S&P500 Technical Indicators",
        description="Common indicators for analyzing market behavior",
        ta=[
            {"kind": "rsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"kind": "bbands", "length": 20, "std": 2},
            {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
            {"kind": "atr", "length": 14},
            {"kind": "sma", "length": 200},
            {"kind": "sma", "length": 50},
            {"kind": "ema", "length": 20},
            {"kind": "cci", "length": 20},
        ]
    )
    
    # Calculate all technical indicators at once
    df.ta.strategy(strat)
    
    # Fix Bollinger Bands naming to match what's expected by the ML model
    # Rename from pandas_ta format (with decimal) to the format expected by ML model
    if 'BBL_20_2.0' in df.columns:
        df['BBL_20_2'] = df['BBL_20_2.0']
    if 'BBM_20_2.0' in df.columns:
        df['BBM_20_2'] = df['BBM_20_2.0']
    if 'BBU_20_2.0' in df.columns:
        df['BBU_20_2'] = df['BBU_20_2.0']
    
    # Calculate Bollinger Band Position (normalized position within Bollinger Bands)
    if all(col in df.columns for col in ['BBL_20_2', 'BBU_20_2']):
        df['BBP_20_2'] = (df['Close'] - df['BBL_20_2']) / (df['BBU_20_2'] - df['BBL_20_2'])
    
    # Handle ATR naming and calculations
    if 'ATRr_14' in df.columns:
        df['ATR_14'] = df['ATRr_14']  # Rename to format expected by ML model
        df['ATR_Pct'] = df['ATR_14'] / df['Close'] * 100
    
    # Volume analysis - make sure names match those in ML model features
    # Calculate average volumes
    df['avg_vol_10'] = df['Volume'].rolling(window=10).mean()
    df['avg_vol_20'] = df['Volume'].rolling(window=20).mean()
    df['avg_vol_50'] = df['Volume'].rolling(window=50).mean()
    
    # Calculate volume ratios (current volume to average)
    df['volume_ratio_10d'] = df['Volume'] / df['avg_vol_10']
    df['volume_ratio_20d'] = df['Volume'] / df['avg_vol_20']
    df['volume_ratio_50d'] = df['Volume'] / df['avg_vol_50']
    
    # Calculate additional volume metrics used by ML features
    df['Volume_ROC_5d'] = df['Volume'].pct_change(periods=5) * 100
    df['Vol_10_50_Ratio'] = df['avg_vol_10'] / df['avg_vol_50']
    
    # Legacy names (keep for backward compatibility)
    df['Volume_Ratio'] = df['volume_ratio_50d']
    df['Avg_Vol_50'] = df['avg_vol_50']
    
    # Calculate price in relation to moving averages
    if 'SMA_50' in df.columns:
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50'] - 1
    if 'SMA_200' in df.columns:
        df['Price_to_SMA200'] = df['Close'] / df['SMA_200'] - 1
    
    # Calculate moving average crossovers
    if all(col in df.columns for col in ['SMA_50', 'SMA_200']):
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
    
    # Add VIX-related features if VIX data is available
    if 'VIX_Close' in df.columns:
        # VIX Moving Averages
        df['VIX_5D_Avg'] = df['VIX_Close'].rolling(window=5).mean()
        df['VIX_20D_Avg'] = df['VIX_Close'].rolling(window=20).mean()
        
        # VIX relative to its averages
        df['VIX_Rel_5D'] = (df['VIX_Close'] / df['VIX_5D_Avg'] - 1) * 100
        df['VIX_Rel_20D'] = (df['VIX_Close'] / df['VIX_20D_Avg'] - 1) * 100
        
        # VIX daily range
        if all(col in df.columns for col in ['VIX_High', 'VIX_Low']):
            df['VIX_HL_Range'] = (df['VIX_High'] - df['VIX_Low']) / df['VIX_Close'] * 100
        
        # VIX return
        df['VIX_Return'] = df['VIX_Close'].pct_change() * 100
    
    # Add rate of decline metrics
    # Calculate rolling sum of negative returns
    df['Rolling_5d_Neg_Returns'] = df['Return'].apply(lambda x: min(x, 0)).rolling(window=5).sum()
    df['Rolling_10d_Neg_Returns'] = df['Return'].apply(lambda x: min(x, 0)).rolling(window=10).sum()
    
    # Count negative days in rolling windows
    df['Rolling_5d_Neg_Days'] = df['Return'].apply(lambda x: 1 if x < 0 else 0).rolling(window=5).sum()
    
    # Current decline rate metrics
    df['Decline_Rate_Per_Day'] = df['Return'].apply(lambda x: x if x < 0 else 0)
    
    # Maximum daily decline in rolling window
    df['Max_Daily_Decline'] = df['Return'].rolling(window=5).min()
    
    # Average decline rate over 5 days (only counting negative days)
    df['Avg_Decline_Rate_5d'] = df['Rolling_5d_Neg_Returns'] / df['Rolling_5d_Neg_Days'].replace(0, np.nan)
    
    # Standard deviation of declines (volatility of declines)
    df['Decline_Volatility_5d'] = df['Return'].apply(lambda x: x if x < 0 else np.nan).rolling(window=5).std()
    
    # Flag for accelerating decline pattern
    # Define accelerating decline as when each day's decline is worse than the previous
    df['Accelerating_Decline'] = df['Return'].rolling(window=3).apply(
        lambda x: 1 if (len(x) == 3 and x[0] < 0 and x[1] < 0 and x[2] < 0 and x[0] > x[1] > x[2]) else 0,
        raw=True
    )
    
    # Drawdown metrics
    # Find running maximum (peak)
    df['Running_Max'] = df['Close'].cummax()
    # Calculate drawdown from peak
    df['Drawdown_From_Peak_Pct'] = (df['Close'] / df['Running_Max'] - 1) * 100
    
    # Calculate days since peak (alternate approach since cumulative_min() is not available)
    not_at_peak = (df['Close'] != df['Running_Max']).astype(int)
    # Reset counter when we hit a new peak
    reset_points = (not_at_peak.diff() < 0).fillna(0).astype(int)
    # Calculate group ID for each peak-to-trough sequence
    group_id = reset_points.cumsum()
    # Count days within each group
    df['Days_Since_Peak'] = not_at_peak.groupby(group_id).cumsum()
    
    # Calculate rate of decline from peak
    df['Peak_To_End_Rate'] = df['Drawdown_From_Peak_Pct'] / df['Days_Since_Peak'].replace(0, 1)
    
    # Calculate percentage of drawdown (useful for understanding where we are in drawdown cycle)
    df['Max_Drawdown_20D'] = df['Drawdown_From_Peak_Pct'].rolling(window=20).min()
    df['Pct_Of_Drawdown'] = df['Drawdown_From_Peak_Pct'] / df['Max_Drawdown_20D'].replace(0, np.nan) * 100
    
    # Set decline duration (for single day events this is 1)
    df['Decline_Duration'] = (df['Return'] < 0).astype(int)
    
    # Price-Volume Correlation (rolling 10-day correlation between price changes and volume)
    df['Price_Volume_Correlation'] = df['Return'].rolling(10).corr(df['Volume'])
    
    # Volume Momentum (acceleration in volume)
    df['Volume_Momentum'] = df['Volume'].pct_change().rolling(5).mean() * 100
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Return']) * df['Volume']).cumsum()
    
    # Volume-Weighted Average Price (VWAP) Ratio
    # Simplified calculation using daily data
    df['Daily_VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
    df['Cum_VWAP'] = df['Daily_VWAP'].rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['VWAP_Ratio'] = df['Close'] / df['Cum_VWAP']
    
    # Clean up NaN values for essential indicators
    df = df.dropna(subset=['RSI_14', 'MACDh_12_26_9'])
    
    return df

def get_indicator_explanation(indicator_name, value):
    """
    Get explanation for technical indicators
    
    Parameters:
    -----------
    indicator_name : str
        Name of the indicator
    value : float
        Current value of the indicator
        
    Returns:
    --------
    str
        Explanation of the indicator's current value
    """
    explanations = {
        'RSI_14': {
            'title': 'Relative Strength Index (14)',
            'bullish': 'Oversold condition, potential bullish reversal',
            'bearish': 'Overbought condition, potential bearish reversal',
            'neutral': 'Neutral momentum conditions',
            'threshold_bullish': 30,
            'threshold_bearish': 70
        },
        'STOCHk_14_3_3': {
            'title': 'Stochastic Oscillator %K (14,3,3)',
            'bullish': 'Oversold condition, watch for crossover signals',
            'bearish': 'Overbought condition, potential resistance ahead',
            'neutral': 'Neutral momentum conditions',
            'threshold_bullish': 20,
            'threshold_bearish': 80
        },
        'BBP_20_2': {
            'title': 'Bollinger Band Position',
            'bullish': 'Price near lower band, potential support',
            'bearish': 'Price near upper band, potential resistance',
            'neutral': 'Price within middle of bands, neutral conditions',
            'threshold_bullish': 0.2,
            'threshold_bearish': 0.8
        },
        'MACDh_12_26_9': {
            'title': 'MACD Histogram (12,26,9)',
            'bullish': 'Positive and increasing, strong bullish momentum',
            'bearish': 'Negative and decreasing, strong bearish momentum',
            'neutral': 'Near zero line, consolidation or trend change possible',
            'threshold_bullish': 0,
            'threshold_bearish': 0,
            'reverse': True
        },
        'ATR_Pct': {
            'title': 'Average True Range %',
            'bullish': 'High volatility, potential for significant price movements',
            'bearish': 'High volatility, potential for significant price movements',
            'neutral': 'Low volatility, range-bound conditions likely',
            'threshold_bullish': 2.0,
            'threshold_bearish': 2.0,
            'no_direction': True
        },
        'Volume_Ratio': {
            'title': 'Volume Ratio (Current/50-day Avg)',
            'bullish': 'Above average volume, strong confirmation of price movement',
            'bearish': 'Above average volume, strong confirmation of price movement',
            'neutral': 'Below average volume, weak price movement',
            'threshold_bullish': 1.5,
            'threshold_bearish': 1.5,
            'no_direction': True
        }
    }
    
    if indicator_name not in explanations:
        return {
            'title': indicator_name,
            'status': 'neutral',
            'explanation': 'No explanation available for this indicator.'
        }
    
    ind = explanations[indicator_name]
    
    if ind.get('no_direction', False):
        # Indicators where high values don't have direction (like volatility)
        status = 'bullish' if value > ind['threshold_bullish'] else 'neutral'
        explanation = ind['bullish'] if value > ind['threshold_bullish'] else ind['neutral']
    elif ind.get('reverse', False):
        # Indicators where the logic is reversed
        if value > ind['threshold_bearish']:
            status = 'bullish'
            explanation = ind['bullish']
        elif value < ind['threshold_bullish']:
            status = 'bearish'
            explanation = ind['bearish']
        else:
            status = 'neutral'
            explanation = ind['neutral']
    else:
        # Standard indicators
        if value < ind['threshold_bullish']:
            status = 'bullish'
            explanation = ind['bullish']
        elif value > ind['threshold_bearish']:
            status = 'bearish'
            explanation = ind['bearish']
        else:
            status = 'neutral'
            explanation = ind['neutral']
    
    return {
        'title': ind['title'],
        'status': status,
        'explanation': explanation
    }
