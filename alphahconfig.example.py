# alphahconfig.example.py — copy to alphahconfig.py and fill in your own values.
# The real alphahconfig.py is gitignored (holds live secrets). Only ALPACA_* + OLLAMA_BASE_URL
# are required for the core system (QC4 + momentum + local-LLM Oracle); the rest are optional.

# alphahconfig.py

# ==========================================
# SYSTEM & SCHEDULING
# ==========================================
RUN_INTERVAL_MINUTES = 15           # How often the background process runs
LIBRARY_FILE = "library.json"       # Flat file for stock tracking
LOG_FILE = "actions.log"            # Timestamped log for all system actions
DEBUG_MODE = True

# ==========================================
# REDDIT SCRAPING SETTINGS (No API Keys Needed)
# ==========================================
#REDDIT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 PsychoAlpha/0.1"
REDDIT_USER_AGENT = "REPLACE_WITH_YOUR_REDDIT_USER_AGENT"
SUBREDDITS = ["pennystocks", "pennystock", "stockstobuytoday"]
POST_LIMIT_PER_SUB = 50             # Number of hot/new posts to scrape per run

# Reddit OAuth (reliable route — bypasses bot-blocking that kills bare .json/.rss scraping).
# Create a "script" app at https://www.reddit.com/prefs/apps  -> fill these 4:
REDDIT_CLIENT_ID = "REPLACE_WITH_YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "REPLACE_WITH_YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USERNAME = "REPLACE_WITH_YOUR_REDDIT_USERNAME"
REDDIT_PASSWORD = "REPLACE_WITH_YOUR_REDDIT_PASSWORD"
MIN_MENTIONS_TO_ADD = 1             # Ticker must be mentioned this many times to enter the library
MAX_POST_AGE_MINUTES = 1000

# ==========================================
# LLM PROVIDER & AGENT SETTINGS
# ==========================================
# Using Local Models via Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# Agent 1: Social Media Analyst
SOCIAL_AGENT_PROVIDER = "ollama"
SOCIAL_AGENT_MODEL = "llama3.1:latest"
#SOCIAL_AGENT_MODEL = "qwen2.5:32b"


# Agent 2: Market Analyst
MARKET_AGENT_PROVIDER = "ollama"
MARKET_AGENT_MODEL = "llama3.1:latest"

# Agent 3: The Money Agent
MONEY_AGENT_PROVIDER = "ollama"
MONEY_AGENT_MODEL = "gemma2:9b"

# Agent 4: News Scout
# Swap between models to compare grading quality vs speed:
#   "llama3.1:latest"  -> faster, lighter, good baseline
#   "qwen3:30b"        -> slower, heavier, better article comprehension
NEWS_AGENT_PROVIDER = "ollama"
NEWS_AGENT_MODEL = "qwen2.5:32b"

# ==========================================
# AGENT 2: MARKET ANALYST SETTINGS
# ==========================================
MARKET_LOOKBACK_DAYS = 60           # Configurable timeframe (30, 60, 90)
INCLUDE_INDICATORS = [
    "SMA_20", "SMA_50",             # Simple Moving Averages
    "RSI_14",                       # Relative Strength Index
    "MACD",                         # Moving Average Convergence Divergence
    "VOLUME_AVG_10"                 # 10-day volume average
]

# ==========================================
# AGENT 3: THE MONEY AGENT (RISK MANAGEMENT)
# ==========================================
DEFAULT_PROFIT_TARGET_PCT = 0.05    # Default +5% profit target 
DEFAULT_STOP_LOSS_PCT = 0.02        # Default -2% stop loss
MAX_RISK_PER_TRADE_USD = 500        # Theoretical max allocation per play
GEMINI_API_KEY = "REPLACE_WITH_YOUR_GEMINI_API_KEY"
# Toggle between "ollama" (local, slow) and "gemini" (cloud, fast)
MONEY_AGENT_PROVIDER = "ollama"

# ==========================================
# AGENT 4: NEWS SCOUT SETTINGS
# ==========================================
NEWS_RATE_LIMIT_SECONDS = 0.5       # Sleep between RSS calls (respect Google)
NEWS_MAX_ARTICLES_PER_TICKER = 3    # Max articles to grade per ticker per run
NEWS_SCORE_THRESHOLD = 70           # Minimum score to include in ranked output
NEWS_TOP_N_PICKS = 10               # How many candidates to validate via yfinance
NEWS_UNIVERSE_FILE = "valid_tickers_cache.json"
 
# News fetch window (mirrors backtester.py logic):
# Prior day 4pm close -> target day 9am pre-market
# Only the freshest overnight catalysts qualify
NEWS_WINDOW_START_HOUR = 16         # Prior trading day close
NEWS_WINDOW_END_HOUR = 9            # Target day pre-market cutoff
 
# Toxic keywords — score zeroed immediately, no LLM call wasted
NEWS_TOXIC_KEYWORDS = [
    "dilution", "offering", "shelf registration",
    "reverse split", "warrant", "bankruptcy",
    "sec investigation", "fraud", "delisted"
]

# ==========================================
# AGENT 5: THE QUANTITATIVE RESEARCHER
# ==========================================
QUANT_AGENT_PROVIDER = "ollama"
QUANT_AGENT_MODEL = "qwen2.5:32b"
QUANT_MAX_ITERATIONS = 50
# Honest in-sample (is_days.json) baseline for ORB validated params after the
# 2026-06-16 timezone-leak fix. Agent must beat this on in-sample, then survive OOS.
QUANT_BASELINE_PNL = -2538.33


# ==========================================
# SMS ALERT CONFIGURATION
# ==========================================
GMAIL_USER = "REPLACE_WITH_YOUR_GMAIL_USER"
GMAIL_APP_PASSWORD = "REPLACE_WITH_YOUR_GMAIL_APP_PASSWORD"
PHONE_GATEWAY = "REPLACE_WITH_YOUR_PHONE_GATEWAY"


# ==========================================
# ALPACA CREDS
# ==========================================
ALPACA_KEY_ID = "REPLACE_WITH_YOUR_ALPACA_KEY_ID"
ALPACA_SECRET_KEY = "REPLACE_WITH_YOUR_ALPACA_SECRET_KEY"


# ==========================================
# INTERACTIVE RESEARCH AGENT
# ==========================================
RESEARCH_AGENT_BACKEND = "ollama"          # "ollama" (local qwen) or "anthropic" (cloud)
RESEARCH_AGENT_MODEL = "qwen2.5:32b"       # used when backend == ollama
ANTHROPIC_API_KEY = "REPLACE_WITH_YOUR_ANTHROPIC_API_KEY"
ANTHROPIC_MODEL = "claude-sonnet-4-6"       # used when backend == anthropic
TAVILY_API_KEY = "REPLACE_WITH_YOUR_TAVILY_API_KEY"

ALERT_EMAIL = "REPLACE_WITH_YOUR_ALERT_EMAIL"   # inbox for Oracle verdict emails
