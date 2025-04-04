import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Check if the DataFrame has a MultiIndex
def debug_data_structure():
    # Fetch some data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Fetch data
    sp500 = yf.download(
        "^GSPC",
        start=start_date,
        end=end_date,
        progress=False
    )
    
    # Print info about the DataFrame
    print("DataFrame info:")
    print(f"Index type: {type(sp500.index)}")
    print(f"Columns: {sp500.columns}")
    print(f"Sample data head:\n{sp500.head()}")
    
    # Check if this is a MultiIndex
    print(f"Is MultiIndex: {isinstance(sp500.index, pd.MultiIndex)}")
    print(f"Is DatetimeIndex: {isinstance(sp500.index, pd.DatetimeIndex)}")
    
if __name__ == "__main__":
    debug_data_structure()