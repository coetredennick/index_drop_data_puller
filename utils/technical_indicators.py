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
    
    # Calculate Bollinger Band Position (normalized position within Bollinger Bands)
    if all(col in df.columns for col in ['BBL_20_2.0', 'BBU_20_2.0']):
        df['BBP_20_2'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    # Calculate ATR as percentage of price
    if 'ATRr_14' in df.columns:
        df['ATR_Pct'] = df['ATRr_14'] / df['Close'] * 100
    
    # Volume analysis
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=50).mean()
    df['Avg_Vol_50'] = df['Volume'].rolling(window=50).mean()
    
    # Calculate price in relation to moving averages
    if 'SMA_50' in df.columns:
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50'] - 1
    if 'SMA_200' in df.columns:
        df['Price_to_SMA200'] = df['Close'] / df['SMA_200'] - 1
    
    # Calculate moving average crossovers
    if all(col in df.columns for col in ['SMA_50', 'SMA_200']):
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
    
    # Clean up NaN values
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
