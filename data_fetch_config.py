"""
Configuration file for data fetching settings to handle Yahoo Finance rate limits
"""

# Default date range (use 2 years by default to avoid rate limits)
DEFAULT_DAYS = 730  # 2 years

# API request delays (in seconds)
INITIAL_DELAY = 3  # Initial delay before starting fetch
BETWEEN_INDICES_DELAY = 15  # Delay between fetching different indices
BETWEEN_CHUNKS_DELAY = 10  # Delay between fetching chunks
VOLATILITY_FETCH_DELAY = 8  # Delay before fetching volatility data

# Rate limit handling
MAX_RETRIES = 5  # Maximum number of retry attempts
BASE_RETRY_DELAY = 10  # Base delay for retry exponential backoff
MAX_RETRY_DELAY = 60  # Maximum delay between retries

# Chunking settings
CHUNK_SIZE_DAYS = 365  # Size of each chunk when fetching long periods
MINI_CHUNK_SIZE_DAYS = 90  # Size of mini-chunks for very large requests

# Cache settings
CACHE_TTL = 1800  # Cache time-to-live in seconds (30 minutes)