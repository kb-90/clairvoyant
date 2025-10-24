import argparse
import ccxt
import pandas as pd
import numpy as np
import joblib
import time
import os
from pathlib import Path
import asyncio
import logging
from typing import List
from dotenv import load_dotenv
# Import TA and sentiment libraries
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator
import nltk
from afinn import Afinn
from lexicon.crypto_lexicon import CRYPTO_LEXICON
import matplotlib.pyplot as plt
import feedparser
import aiohttp
import seaborn as sns
import io
import re

# Import ML/DL libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, Bidirectional, Conv1D, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard # type: ignore
from lightgbm import LGBMRegressor
import xgboost as xgb

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# --- TERMINAL STYLING (ENHANCED) ---
class TerminalColors:
    """ANSI color codes for terminal styling"""
    HEADER = '\033[95m'      # Magenta/Purple
    OKBLUE = '\033[94m'      # Blue
    OKCYAN = '\033[96m'      # Cyan
    OKGREEN = '\033[92m'     # Green
    WARNING = '\033[93m'     # Yellow/Orange
    FAIL = '\033[91m'        # Red
    ENDC = '\033[0m'         # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    # Custom colors for enhanced theme
    TURQUOISE = '\033[38;5;51m'
    PURPLE = '\033[38;5;141m'
    PINK = '\033[38;5;213m'
    ORANGE = '\033[38;5;208m'
    
class TerminalStyle:
    """Enhanced terminal output with modern, trader-friendly design"""
    
    @staticmethod
    def header(text):
        """Main section header with gradient effect"""
        width = 80
        print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * width}{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{text.center(width-2)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * width}{TerminalColors.ENDC}\n")
    
    @staticmethod
    def subheader(text):
        """Subsection header"""
        print(f"\n{TerminalColors.BOLD}{TerminalColors.PURPLE}▶ {text}{TerminalColors.ENDC}")
        print(f"{TerminalColors.DIM}{TerminalColors.TURQUOISE}{'─' * 60}{TerminalColors.ENDC}")
    
    @staticmethod
    def success(text):
        """Success message"""
        print(f"{TerminalColors.OKGREEN}✓ {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def info(text):
        """Info message"""
        print(f"{TerminalColors.OKCYAN}● {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def warning(text):
        """Warning message"""
        print(f"{TerminalColors.ORANGE}⚠ {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def error(text):
        """Error message"""
        print(f"{TerminalColors.FAIL}✗ {text}{TerminalColors.ENDC}")
    
    @staticmethod
    def metric(label, value, unit="", positive=None):
        """Display a metric with optional color coding"""
        color = TerminalColors.ENDC
        if positive is True:
            color = TerminalColors.OKGREEN
        elif positive is False:
            color = TerminalColors.FAIL
        print(f"  {TerminalColors.DIM}{label}:{TerminalColors.ENDC} {color}{TerminalColors.BOLD}{value}{unit}{TerminalColors.ENDC}")
    
    @staticmethod
    def prediction_box(current, predicted, change_pct, horizon, confidence_lower=None, confidence_upper=None, confidence_score=None):
        """Enhanced prediction display with visual indicators"""
        color = TerminalColors.OKGREEN if change_pct > 0 else TerminalColors.FAIL
        symbol = "▲" if change_pct > 0 else "▼"
        width = 58

        # Helper to strip ANSI codes for accurate length calculation
        def strip_ansi(text):
            return re.sub(r'\x1B(?:[@-Z\-_]|[[0-?]*[ -/]*[@-~])', '', text)

        # Confidence visualization
        conf_color = TerminalColors.ENDC
        conf_bar = ""
        if confidence_score is not None:
            bar_length = int(confidence_score / 5)
            if confidence_score >= 75:
                conf_color = TerminalColors.OKGREEN
                conf_bar = "█" * bar_length
            elif confidence_score >= 50:
                conf_color = TerminalColors.ORANGE
                conf_bar = "▓" * bar_length
            else:
                conf_color = TerminalColors.FAIL
                conf_bar = "░" * bar_length
        
        def print_line(content):
            padding = width - len(strip_ansi(content))
            print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}{content}{' ' * padding}{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")

        print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╔{'═' * width}╗{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{f'  {horizon}H FORECAST'.center(width)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╠{'═' * width}╣{TerminalColors.ENDC}")
        
        print_line(f"  Current Price:   {TerminalColors.BOLD}{current:>12.4f}{TerminalColors.ENDC} USDT")
        print_line(f"  Predicted Price: {TerminalColors.BOLD}{predicted:>12.4f}{TerminalColors.ENDC} USDT")
        
        if confidence_lower is not None and confidence_upper is not None:
            print_line(f"  Range:           {TerminalColors.DIM}{confidence_lower:.4f} - {confidence_upper:.4f}{TerminalColors.ENDC}")
        
        print_line(f"  Expected Change: {color}{TerminalColors.BOLD}{symbol} {abs(change_pct):>10.2f}%{TerminalColors.ENDC}")
        
        if confidence_score is not None:
            print_line(f"  Confidence:      {conf_color}{conf_bar} {TerminalColors.BOLD}{confidence_score:>5.1f}%{TerminalColors.ENDC}")
        
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}╚{'═' * width}╝{TerminalColors.ENDC}")
    
    @staticmethod
    def progress(current, total, task):
        """Enhanced progress indicator"""
        percentage = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = f"{TerminalColors.PINK}{'█' * filled}{TerminalColors.DIM}{'░' * (bar_length - filled)}{TerminalColors.ENDC}"
        print(f"\r{TerminalColors.TURQUOISE}{task}: {bar} {TerminalColors.BOLD}{percentage:.0f}%{TerminalColors.ENDC}", end='', flush=True)
        if current == total:
            print()

# --- 0. CONFIGURATION & SETUP ---
load_dotenv()
MODEL_DIR = Path("models")

# CONFIDENCE INTERVAL CONFIGURATION
CI_METHOD = 'adaptive'

# --- TRAINING CONFIGURATION ---
# --- TRAINING CONFIGURATION ---
# Parameters are now loaded from the .env file.
# Default values are provided as a fallback.
TRAINING_CONFIG = {
    # adjust in .env
    "TICKER": os.getenv("TICKER", "XRP/USDT"),
    # adjust in .env
    "TIMEFRAME": os.getenv("TIMEFRAME", "1h"),
    # adjust in .env - Comma-separated list of integers
    "PREDICTION_HORIZONS": [int(h) for h in os.getenv("PREDICTION_HORIZONS", "48").split(',')],
    # adjust in .env
    "SEQUENCE_LENGTH": int(os.getenv("SEQUENCE_LENGTH", 60)),
    # adjust in .env
    "DATA_LIMIT": int(os.getenv("DATA_LIMIT", 5000)),
    # adjust in .env
    "CV_EPOCHS_DL": int(os.getenv("CV_EPOCHS_DL", 50)),
    # adjust in .env
    "CV_PATIENCE": int(os.getenv("CV_PATIENCE", 8)),
    # adjust in .env
    "DL_EPOCHS": int(os.getenv("DL_EPOCHS", 50)),
    # adjust in .env - Must be "True" or "False"
    "OPTIMIZE_HYPERPARAMETERS": os.getenv("OPTIMIZE_HYPERPARAMETERS", "False").lower() in ('true', '1', 't'),
    # adjust in .env
    "OPTUNA_TRIALS": int(os.getenv("OPTUNA_TRIALS", 50)),
    
    # --- Model-specific parameters (remain in script for clarity) ---
    "MODEL_PARAMS": {
        "rf": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "random_state": 42, "n_jobs": -1},
        "lgbm": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 10, "random_state": 42, "verbose": -1, "n_jobs": -1},
        "xgb": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 10, "random_state": 42, "tree_method": "hist", "n_jobs": -1},
        "gbm": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 8, "random_state": 42},
    }
}

# Import Optuna for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
    def create_optuna_study(model_name: str) -> optuna.study.Study:
        """Creates an Optuna study with a SQLite backend."""
        storage_path = f"sqlite:///dtbs/{model_name}_optimization.db"
        return optuna.create_study(
            storage=storage_path,
            study_name=f"{model_name}_optimization",
            load_if_exists=True,
            direction='minimize'
        )
except ImportError:
    OPTUNA_AVAILABLE = False
    if TRAINING_CONFIG["OPTIMIZE_HYPERPARAMETERS"]:
        TerminalStyle.warning("Optuna not available. Install with: pip install optuna")
        TerminalStyle.info("Disabling hyperparameter optimization")
        TRAINING_CONFIG["OPTIMIZE_HYPERPARAMETERS"] = False

# Set up custom logging formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    FORMATS = {
        logging.DEBUG: f"{TerminalColors.DIM}%(message)s{TerminalColors.ENDC}",
        logging.INFO: f"{TerminalColors.OKCYAN}● %(message)s{TerminalColors.ENDC}",
        logging.WARNING: f"{TerminalColors.ORANGE}⚠ %(message)s{TerminalColors.ENDC}",
        logging.ERROR: f"{TerminalColors.FAIL}✗ %(message)s{TerminalColors.ENDC}",
        logging.CRITICAL: f"{TerminalColors.FAIL}{TerminalColors.BOLD}✗ %(message)s{TerminalColors.ENDC}"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger('clairvoyant')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

# One-time download for NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    TerminalStyle.info("Downloading required NLTK data (punkt)...")
    nltk.download('punkt', quiet=True)
    TerminalStyle.success("NLTK data downloaded")

# --- 1. DYNAMIC ON-CHAIN ANALYSIS ENGINE ---

async def fetch_top_xrp_accounts(session: aiohttp.ClientSession, limit: int = 50) -> List[str]:
    """
    Fetches the top XRP accounts from the xrpscan.com API.
    """
    url = "https://api.xrpscan.com/api/v1/balances"
    TerminalStyle.info(f"Fetching top {limit} whale accounts from xrpscan API")
    headers = {'Accept': 'application/json'}
    try:
        async with session.get(url, timeout=30, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                addresses = [item['account'] for item in data]
                if addresses:
                    TerminalStyle.success(f"Dynamically found {len(addresses)} unique whale accounts.")
                    return addresses[:limit]
                logger.warning("Could not parse any addresses from the xrpscan API response.")
                return []
            logger.warning(f"Failed to fetch rich list from xrpscan API, status: {response.status}")
            return []
    except Exception as e:
        logger.error(f"Error fetching dynamic accounts from xrpscan API: {e}")
        return []

class XRPOnChainAnalyzer:
    """
    Analyzes on-chain metrics for a dynamic list of top XRP accounts.
    """
    def __init__(self):
        self.base_url = "https://xrplcluster.com"
        self.semaphore = asyncio.Semaphore(2)  # OPTIMIZED: Increased from 1 to 2

    async def _fetch_json_rpc(self, session: aiohttp.ClientSession, method: str, params: list = None) -> dict:
        """Sends a JSON-RPC request with rate-limiting and retry logic."""
        payload = {"method": method, "params": params or [{}]}
        max_retries = 1  # Reduced retries as batching should prevent most errors
        base_delay = 5.0 # Increased delay for the single retry

        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    async with session.post(self.base_url, json=payload, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'result' in data and data['result'].get('status') == 'success':
                                return data['result']
                            else:
                                error = data.get('result', {}).get('error_message', 'Unknown RPC error')
                                logger.warning(f"XRP RPC '{method}' failed: {error}")
                                return {}
                        elif response.status == 429:
                            if attempt < max_retries:
                                delay = base_delay * (2 ** attempt)
                                logger.warning(f"Rate limit hit for {method}. Retrying in {delay:.1f}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Rate limit hit for {method}. Max retries exceeded.")
                                return {}
                        else:
                            logger.warning(f"XRP RPC returned status {response.status} for {method}")
                            return {}
                except Exception as e:
                    logger.warning(f"Error during RPC call for {method}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(base_delay)
                    else:
                        logger.error(f"Failed to fetch from XRP RPC after multiple retries: {e}")
                        return {}
        return {}

    async def fetch_whale_transactions(self, session: aiohttp.ClientSession, top_accounts: List[str], threshold_xrp: float = 500_000) -> pd.DataFrame:
        """
        Fetches large transactions from a provided list of top accounts over the last 6 hours using a batching approach.
        """
        if not top_accounts:
            TerminalStyle.warning("No dynamic accounts provided for whale tracking.")
            return pd.DataFrame()

        TerminalStyle.info(f"Scanning for whale activity across {len(top_accounts)} largest accounts (last 6 hours)...")
        
        ledger_current = await self._fetch_json_rpc(session, "ledger_current")
        if not ledger_current or 'ledger_current_index' not in ledger_current:
            TerminalStyle.warning("Could not fetch current ledger index for whale scan.")
            return pd.DataFrame()
        
        current_index = ledger_current['ledger_current_index']
        # Approx. 15 ledgers per minute -> 6 * 60 * 15 = 5400 ledgers
        ledgers_in_6h = 5400
        start_index = current_index - ledgers_in_6h
        
        all_results = []
        batch_size = 5  # Reduced batch size to be more conservative
        wait_time = 10  # seconds between batches

        for i in range(0, len(top_accounts), batch_size):
            batch_accounts = top_accounts[i:i + batch_size]
            tasks = []
            # Corrected batch numbering to be 1-based
            total_batches = (len(top_accounts) + batch_size - 1) // batch_size
            TerminalStyle.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_accounts)} accounts)...")
            for wallet_address in batch_accounts:
                params = [{"account": wallet_address, "ledger_index_min": start_index, "ledger_index_max": current_index, "limit": 50, "forward": True}]
                tasks.append(self._fetch_json_rpc(session, "account_tx", params))
            
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            if i + batch_size < len(top_accounts):
                TerminalStyle.info(f"Waiting {wait_time}s before next batch to respect API limits...")
                await asyncio.sleep(wait_time)

        whale_metrics = []
        processed_tx_hashes = set()
        for account_txs in all_results:
            if not account_txs or 'transactions' not in account_txs: continue
            
            for tx_wrapper in account_txs['transactions']:
                tx = tx_wrapper.get('tx', {})
                tx_hash = tx.get('hash')
                if not tx_hash or tx_hash in processed_tx_hashes: continue

                if tx.get('TransactionType') == 'Payment' and isinstance(tx.get('Amount'), str):
                    amount_xrp = int(tx['Amount']) / 1_000_000
                    if amount_xrp >= threshold_xrp:
                        processed_tx_hashes.add(tx_hash)
                        timestamp = pd.to_datetime(tx.get('date', 0) + 946684800, unit='s', utc=True)
                        whale_metrics.append({'timestamp': timestamp, 'whale_amount': amount_xrp, 'whale_count': 1})

        if not whale_metrics:
            TerminalStyle.warning("No significant whale transactions found in the last 6 hours.")
            return pd.DataFrame()

        df = pd.DataFrame(whale_metrics).groupby('timestamp').agg({'whale_amount': 'sum', 'whale_count': 'sum'}).reset_index()
        total_xrp = df['whale_amount'].sum()
        total_count = df['whale_count'].sum()
        TerminalStyle.success(f"Detected {total_count:.0f} whale txs totaling {total_xrp:,.2f} XRP in the last 6 hours.")
        return df

async def integrate_onchain_metrics(df: pd.DataFrame, ticker_symbol: str = 'XRP') -> tuple:
    """
    Fetches and integrates on-chain data from a dynamic list of top XRP accounts.
    """
    if ticker_symbol != 'XRP':
        TerminalStyle.warning(f"On-chain analysis currently only supports XRP, skipping for {ticker_symbol}")
        return df, pd.DataFrame()
    
    TerminalStyle.subheader("Phase 1b: On-Chain Metrics Analysis (Dynamic)")
    whale_df = pd.DataFrame()
    try:
        async with aiohttp.ClientSession() as session:
            top_accounts = await fetch_top_xrp_accounts(session, limit=50)
            
            if not top_accounts:
                TerminalStyle.error("Could not fetch top accounts. Skipping on-chain analysis.")
                df['whale_volume_xrp'] = 0.0
                df['whale_tx_count'] = 0.0
                return df, whale_df

            analyzer = XRPOnChainAnalyzer()
            whale_df = await analyzer.fetch_whale_transactions(session, top_accounts=top_accounts)

        if not whale_df.empty:
            whale_hourly = whale_df.set_index('timestamp').resample('h').agg({'whale_amount': 'sum', 'whale_count': 'sum'}).rename(columns={'whale_amount': 'whale_volume_xrp', 'whale_count': 'whale_tx_count'})
            df = df.join(whale_hourly, how='left')
            for col in ['whale_volume_xrp', 'whale_tx_count']:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0)
            TerminalStyle.success("Dynamic on-chain metrics integrated successfully.")
        else:
            TerminalStyle.warning("No on-chain data was fetched. Continuing without it.")
            df['whale_volume_xrp'] = 0.0
            df['whale_tx_count'] = 0.0
            
    except Exception as e:
        logger.error(f"Failed to fetch or integrate dynamic on-chain metrics: {e}")
        TerminalStyle.warning("Continuing without on-chain data.")
        df['whale_volume_xrp'] = 0.0
        df['whale_tx_count'] = 0.0
    
    return df, whale_df

# --- 2. ENHANCED SENTIMENT ANALYSIS ENGINE ---

def calculate_sentiment(text: str, afinn: Afinn) -> float:
    """
    Calculates a word-level sentiment score using AFINN + crypto lexicon.
    """
    words = text.lower().split()
    total_score, relevant_words = 0, 0
    
    for word in words:
        if word in CRYPTO_LEXICON:
            total_score += CRYPTO_LEXICON[word]
            relevant_words += 1
        elif word in afinn._dict:
            total_score += afinn._dict[word]
            relevant_words += 1
    
    return total_score / relevant_words if relevant_words > 0 else 0.0

def analyze_sentiment(text: str) -> float:
    """
    Analyzes sentiment using AFINN + crypto lexicon.
    """
    chunks = [chunk.strip() for chunk in text.replace('.', ',').split(',') if chunk.strip()]
    if not chunks:
        return 0.0
    
    afinn = Afinn()
    chunk_scores = []
    
    for chunk in chunks:
        score = calculate_sentiment(chunk, afinn)
        chunk_scores.append(score)
        
    final_score = np.mean(chunk_scores)
    return np.clip(final_score / 5.0, -1.0, 1.0)

# --- ENHANCED RSS PARSING ---
async def fetch_feed(session, url):
    """
    Asynchronously fetches and parses a single RSS feed.
    """
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                content = await response.text()
                return feedparser.parse(content)
            logger.warning(f"Failed to fetch {url}, status: {response.status}")
            return None
    except asyncio.TimeoutError:
        logger.warning(f"Timeout error fetching RSS feed from {url}")
        return None
    except aiohttp.ClientError as e:
        logger.warning(f"Client error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching {url}: {e}")
        return None

async def scrape_financial_articles() -> List[dict]:
    """
    Fetches financial news articles from a predefined list of RSS feeds.
    """
    rss_urls = [
        'https://cointelegraph.com/rss',
        'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml',
        'https://decrypt.co/feed',
        'https://crypto.news/feed/',
        'https://u.today/rss',
        'https://bitcoinist.com/category/ripple/feed/',
        'https://dailyhodl.com/ripple-and-xrp/feed/',
        'https://ambcrypto.com/tag/ripple/feed/',
        'https://www.newsbtc.com/feed/',
        'https://cryptobriefing.com/feed/',
        'https://beincrypto.com/feed/',
        'https://coingape.com/feed/',
        'https://cryptonews.com/news/feed/',
        'https://rsscrypto.com/feed/'
    ]
    articles = []
    processed_titles = set()
    twenty_four_hours_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)

    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}) as session:
        tasks = [fetch_feed(session, url) for url in rss_urls]
        feeds = await asyncio.gather(*tasks)

    for feed in feeds:
        if not feed:
            continue
        for entry in feed.entries[:30]:
            title = entry.get('title', '')
            pub_date = entry.get('published', time.strftime('%a, %d %b %Y %H:%M:%S %z', time.gmtime()))
            timestamp = pd.to_datetime(pub_date, utc=True, errors='coerce') or pd.Timestamp.now(tz='UTC')

            if title and title not in processed_titles and timestamp >= twenty_four_hours_ago:
                processed_titles.add(title)
                summary = entry.get('summary', entry.get('description', ''))
                full_text = f"{title}. {summary}"
                articles.append({'text': full_text, 'timestamp': timestamp, 'title': title})

    TerminalStyle.success(f"Found {len(articles)} unique articles from the last 24 hours")
    return articles

def analyze_news_sentiment(articles: List[dict], ticker_symbol: str) -> pd.DataFrame:
    """
    Analyzes and scores news articles for a specific ticker, applying weights.
    """
    all_scores = []
    now = pd.Timestamp.now(tz='UTC')
    ticker_lower = ticker_symbol.lower()

    for article in articles:
        text = article.get('text', '')

        if ticker_lower not in text.lower():
            continue

        score = analyze_sentiment(text)
        timestamp = article['timestamp']
        
        age_hours = (now - timestamp).total_seconds() / 3600
        
        if age_hours <= 3:
            weight = 1.5
        elif age_hours <= 6:
            weight = 1.2
        elif age_hours <= 12:
            weight = 1.0
        elif age_hours <= 24:
            weight = 0.8
        elif age_hours <= 48:
            weight = 0.6
        else:
            weight = 0.4

        if ticker_lower in article.get('title', '').lower():
            weight *= 1.2

        all_scores.append({
            'timestamp': timestamp,
            'sentiment': score,
            'weight': weight
        })

    if all_scores:
        df = pd.DataFrame(all_scores)
        df['weighted_sentiment'] = df['sentiment'] * df['weight']
        
        result = df.groupby('timestamp').agg({
            'weighted_sentiment': 'sum',
            'weight': 'sum'
        })
        
        result['news_sentiment'] = result['weighted_sentiment'] / result['weight']
        result['news_sentiment'] = np.clip(result['news_sentiment'], -1, 1)
        return result[['news_sentiment']].sort_index()
    else:
        return pd.DataFrame()

# --- 3. DATA LOADING & SENTIMENT INTEGRATION ---
def load_data(ticker='XRP/USDT', timeframe='1h', limit=1000):
    """
    Loads historical OHLCV data from Binance with error handling and retries.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    ms_per_unit = exchange.parse_timeframe(timeframe) * 1000
    all_ohlcv = []
    since = exchange.milliseconds() - (limit * ms_per_unit)
    TerminalStyle.info(f"Fetching ~{limit} data points for {ticker} ({timeframe})")
    
    retries = 0
    max_retries = 5
    seen_timestamps = set()  # OPTIMIZED: Track unique timestamps
    
    while len(all_ohlcv) < limit and retries < max_retries:
        try:
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, since=since, limit=1000)
            if not ohlcv:
                logger.warning("No more OHLCV data available from exchange")
                break
            since = ohlcv[-1][0] + 1
            
            # OPTIMIZED: Single-pass deduplication
            for candle in ohlcv:
                ts = candle[0]
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    all_ohlcv.append(candle)
            
            retries = 0
            TerminalStyle.progress(len(all_ohlcv), limit, "Fetching data")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            retries += 1
            logger.warning(f"Network issue (attempt {retries}/{max_retries}): {e}")
            time.sleep(5 * retries)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}. Aborting")
            break
        except Exception as e:
            retries += 1
            logger.error(f"Unexpected error (attempt {retries}/{max_retries}): {e}")
            time.sleep(5 * retries)
    
    TerminalStyle.success(f"Fetched {len(all_ohlcv)} data points")
    if not all_ohlcv:
        logger.error("Could not fetch any OHLCV data. Returning empty DataFrame")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize('UTC')
    df.sort_index(inplace=True)
    return df.tail(limit)

def fetch_and_add_sentiment(df, ticker_symbol='XRP'):
    """
    Fetches and integrates news sentiment data into the main DataFrame.
    """
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    Path('sentiment').mkdir(parents=True, exist_ok=True)
    news_path = Path('sentiment') / f'news_sentiment_{ticker_symbol}.csv'
    
    try:
        news_df = pd.read_csv(news_path, index_col='timestamp', parse_dates=True)
        news_df.index = news_df.index.tz_convert('UTC') if news_df.index.tz is None else news_df.index
        last_cache_time = news_df.index.max()
        if (pd.Timestamp.now(tz='UTC') - last_cache_time) < pd.Timedelta(hours=1):
            TerminalStyle.success("Using cached sentiment data")
            news_df_hourly = news_df.resample('h').mean()
            df = df.join(news_df_hourly, how='left')
            if not df['news_sentiment'].dropna().empty:
                avg_sent = df['news_sentiment'].dropna().mean()
                TerminalStyle.metric("Average Sentiment", f"{avg_sent:+.3f}", positive=(avg_sent > 0))
            df['sentiment'] = df['news_sentiment'].bfill().ffill().fillna(0)
            del df['news_sentiment']
            
            plt.figure(figsize=(10, 5))
            df['sentiment'].tail(24).plot(label='Hourly News Sentiment', marker='o', linestyle='-')
            plt.title(f'{ticker_symbol} Hourly News Sentiment (Last 24h)')
            plt.ylabel('Sentiment Score (-1 to 1)')
            plt.legend()
            plt.grid(True)
            plot_path = Path('sentiment') / f'sentiment_trend_{ticker_symbol}.png'
            plt.savefig(plot_path)
            plt.close()
            TerminalStyle.success(f"Saved sentiment plot to {plot_path}")
            return df
        else:
            TerminalStyle.info("Cache expired, refreshing sentiment data...")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        TerminalStyle.info("No cache found, fetching fresh sentiment data...")
    except Exception as e:
        logger.warning(f"Could not read cache: {e}. Fetching fresh data")

    try:
        try:
            articles = asyncio.run(scrape_financial_articles())
        except RuntimeError as e:
            if 'already running' in str(e):
                loop = asyncio.get_event_loop()
                articles = loop.run_until_complete(scrape_financial_articles())
            else:
                raise
        news_sent_df = analyze_news_sentiment(articles, ticker_symbol)
        if not news_sent_df.empty:
            news_sent_df.to_csv(news_path)
            news_df_hourly = news_sent_df.resample('h').mean()
            df = df.join(news_df_hourly, how='left')
            if not df['news_sentiment'].dropna().empty:
                avg_sent = df['news_sentiment'].dropna().mean()
                TerminalStyle.metric("Average Sentiment", f"{avg_sent:+.3f}", positive=(avg_sent > 0))
            df['sentiment'] = df['news_sentiment'].bfill().ffill().fillna(0)
            del df['news_sentiment']
        else:
            df['sentiment'] = 0.0
            TerminalStyle.warning("No news data found, using neutral sentiment")
    except aiohttp.ClientError as e:
        logger.error(f"Network error during news fetch: {e}. Using neutral sentiment")
        df['sentiment'] = 0.0
    except Exception as e:
        logger.error(f"Unexpected error during news fetch: {e}. Using neutral sentiment")
        df['sentiment'] = 0.0

    plt.figure(figsize=(10, 5))
    df['sentiment'].tail(24).plot(label='Hourly News Sentiment', marker='o', linestyle='-')
    plt.title(f'{ticker_symbol} Hourly News Sentiment (Last 24h)')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.legend()
    plt.grid(True)
    plot_path = Path('sentiment') / f'sentiment_trend_{ticker_symbol}.png'
    plt.savefig(plot_path)
    plt.close()
    TerminalStyle.success(f"Saved sentiment plot to {plot_path}")
    
    return df

# --- 4. FEATURE ENGINEERING ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers a comprehensive set of features for the model.
    """
    if df.empty:
        TerminalStyle.warning("Empty DataFrame provided to feature_engineering. Returning empty DataFrame.")
        return pd.DataFrame()

    required_columns = ['close', 'high', 'low', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        TerminalStyle.warning(f"Missing required columns for feature engineering: {missing_columns}. Returning original DataFrame.")
        return df

    # Momentum Indicators
    rsi = RSIIndicator(df['close'], window=14)
    df['rsi'] = rsi.rsi()

    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    roc = ROCIndicator(df['close'], window=12)
    df['roc'] = roc.roc()

    # Trend Indicators
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    ema12 = EMAIndicator(df['close'], window=12)
    ema26 = EMAIndicator(df['close'], window=26)
    df['ema12'] = ema12.ema_indicator()
    df['ema26'] = ema26.ema_indicator()
    df['ema_cross'] = df['ema12'] - df['ema26']

    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()

    cci = CCIIndicator(df['high'], df['low'], df['close'], window=20)
    df['cci'] = cci.cci()

    # Volatility Indicators
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']

    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()

    kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low'] = kc.keltner_channel_lband()

    # Volume Indicators
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()

    mfi = MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['mfi'] = mfi.money_flow_index()

    cmf = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['cmf'] = cmf.chaikin_money_flow()

    # Price Features
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()

    df.dropna(inplace=True)

    TerminalStyle.success("Feature engineering completed")
    return df

def detect_market_regime(df: pd.DataFrame, log_dir: str = None) -> pd.DataFrame:
    """
    Detects market regimes using K-Means clustering on key features.
    """
    if df.empty:
        TerminalStyle.warning("Empty DataFrame provided to detect_market_regime. Returning empty DataFrame.")
        return pd.DataFrame()

    regime_features = ['rsi', 'macd', 'bb_width', 'atr', 'adx']
    missing_features = [f for f in regime_features if f not in df.columns]
    if missing_features:
        TerminalStyle.warning(f"Missing features for market regime detection: {missing_features}. Skipping regime detection.")
        df['market_regime'] = 0
        return df

    X_regime = df[regime_features].dropna()
    if len(X_regime) < 3:
        TerminalStyle.warning("Insufficient data for market regime detection. Defaulting to neutral regime.")
        df['market_regime'] = 0
        return df

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['market_regime'] = 0
    df.loc[X_regime.index, 'market_regime'] = kmeans.fit_predict(X_regime)

    if log_dir:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[regime_features + ['market_regime']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlations with Market Regimes')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        file_writer = tf.summary.create_file_writer(log_dir)
        with file_writer.as_default():
            tf.summary.image("Market Regime Correlation Heatmap", image, step=0)
        plt.close()

    TerminalStyle.success("Market regime detection completed")
    return df

# --- 5. MODEL ARCHITECTURES & WRAPPERS ---
def get_default_hyperparameters(model_type: str) -> dict:
    """
    Returns default hyperparameters for a given model type.
    """
    if model_type in ['gru', 'lstm']:
        return {
            'n_units': 128,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    elif model_type == 'cnn_lstm':
        return {
            'conv_filters': 64,
            'kernel_size': 3,
            'n_units': 128,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_dl_model(input_shape: tuple, n_units: int = 128, dropout: float = 0.2, model_type: str = 'gru') -> Sequential:
    """
    Creates a deep learning model (GRU or LSTM) with Bidirectional layers.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    RecurrentLayer = GRU if model_type == 'gru' else LSTM
    model.add(Bidirectional(RecurrentLayer(n_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(RecurrentLayer(n_units)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def create_cnn_lstm_model(input_shape: tuple, conv_filters: int = 64, kernel_size: int = 3, n_units: int = 128, dropout: float = 0.2) -> Sequential:
    """
    Creates a CNN-LSTM hybrid model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(LSTM(n_units)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

class StopIfNaN(Callback):
    """Custom callback to stop training if NaN loss is detected."""
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get('loss')
        if loss is not None and np.isnan(loss):
            self.model.stop_training = True
            logger.warning("NaN loss detected. Stopping training.")



import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional
import matplotlib.pyplot as plt
import io

class TradingTensorBoard:
    """
    Simplified TensorBoard logger focused on trader-relevant metrics.
    Organized into clear dashboards: Market, Training, Predictions, Performance
    """
    def __init__(self, log_dir: Path, horizon: int):
        self.log_dir = Path(log_dir)
        self.horizon = horizon
        
        # Create separate writers for different dashboards
        self.writers = {
            'market': tf.summary.create_file_writer(str(self.log_dir / 'market_data')),
            'training': tf.summary.create_file_writer(str(self.log_dir / 'training_metrics')),
            'predictions': tf.summary.create_file_writer(str(self.log_dir / 'prediction_quality')),
            'performance': tf.summary.create_file_writer(str(self.log_dir / 'model_performance'))
        }
        
        self.step = 0
    
    def log_market_context(self, df: pd.DataFrame):
        """Log current market conditions - what traders need to see"""
        with self.writers['market'].as_default():
            # Use last 100 points for context
            recent_df = df.tail(100).reset_index()
            
            for _, row in recent_df.iterrows():
                step = int(row['timestamp'].timestamp())
                
                # Price action
                tf.summary.scalar('Price/Current_USDT', row['close'], step=step)
                
                # Volatility indicators
                if 'atr' in row:
                    tf.summary.scalar('Volatility/ATR', row['atr'], step=step)
                if 'bb_width' in row:
                    tf.summary.scalar('Volatility/Bollinger_Width', row['bb_width'], step=step)
                
                # Momentum
                if 'rsi' in row:
                    tf.summary.scalar('Momentum/RSI', row['rsi'], step=step)
                if 'macd' in row:
                    tf.summary.scalar('Momentum/MACD', row['macd'], step=step)
                
                # Sentiment
                if 'sentiment' in row:
                    tf.summary.scalar('Sentiment/News_Score', row['sentiment'], step=step)
                
                # On-chain (if available)
                if 'whale_volume_xrp' in row and row['whale_volume_xrp'] > 0:
                    tf.summary.scalar('OnChain/Whale_Volume_XRP', row['whale_volume_xrp'], step=step)
                if 'whale_tx_count' in row and row['whale_tx_count'] > 0:
                    tf.summary.scalar('OnChain/Whale_Transactions', row['whale_tx_count'], step=step)
            
            # Market regime distribution
            if 'market_regime' in df.columns:
                regime_counts = df['market_regime'].value_counts()
                for regime, count in regime_counts.items():
                    tf.summary.scalar(f'Market_Regime/Regime_{regime}_Frequency', 
                                    count / len(df) * 100, step=0)
        
        self.writers['market'].flush()
    
    def log_whale_activity(self, whale_df: pd.DataFrame):
        """Log on-chain whale activity in real-time"""
        if whale_df.empty:
            return
            
        with self.writers['market'].as_default():
            # Aggregate metrics
            total_volume = whale_df['whale_amount'].sum()
            total_txs = whale_df['whale_count'].sum()
            avg_tx_size = total_volume / total_txs if total_txs > 0 else 0
            
            tf.summary.scalar('OnChain_Summary/Total_Whale_Volume_6h', total_volume, step=0)
            tf.summary.scalar('OnChain_Summary/Total_Whale_Transactions_6h', total_txs, step=0)
            tf.summary.scalar('OnChain_Summary/Average_Transaction_Size', avg_tx_size, step=0)
            
            # Time series of whale activity
            for _, row in whale_df.iterrows():
                step = int(row['timestamp'].timestamp())
                tf.summary.scalar('OnChain_TimeSeries/Whale_Volume', 
                                row['whale_amount'], step=step)
                tf.summary.scalar('OnChain_TimeSeries/Transaction_Count', 
                                row['whale_count'], step=step)
        
        self.writers['market'].flush()
    
    def log_training_epoch(self, epoch: int, metrics: Dict[str, float], 
                          model_name: str, is_validation: bool = False):
        """Log training metrics per epoch"""
        prefix = 'Validation' if is_validation else 'Training'
        
        with self.writers['training'].as_default():
            # Core training metrics
            if 'loss' in metrics:
                tf.summary.scalar(f'{model_name}/{prefix}_Loss', 
                                metrics['loss'], step=epoch)
            if 'mae' in metrics:
                tf.summary.scalar(f'{model_name}/{prefix}_MAE', 
                                metrics['mae'], step=epoch)
            
            # Learning dynamics
            if 'lr' in metrics:
                tf.summary.scalar(f'{model_name}/Learning_Rate', 
                                metrics['lr'], step=epoch)
        
        self.writers['training'].flush()
    
    def log_cv_fold_results(self, fold: int, metrics: Dict[str, float]):
        """Log cross-validation results"""
        with self.writers['performance'].as_default():
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/RMSE_Fold_{fold}', 
                            metrics['rmse'], step=fold)
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/MAE_Fold_{fold}', 
                            metrics['mae'], step=fold)
            
            if 'directional_accuracy' in metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Direction_Accuracy_Fold_{fold}', 
                                metrics['directional_accuracy'], step=fold)
        
        self.writers['performance'].flush()
    
    def log_final_cv_summary(self, avg_metrics: Dict[str, float]):
        """Log aggregated CV results"""
        with self.writers['performance'].as_default():
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_RMSE', 
                            avg_metrics['rmse'], step=0)
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_MAE', 
                            avg_metrics['mae'], step=0)
            
            if 'directional_accuracy' in avg_metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_Direction_Accuracy', 
                                avg_metrics['directional_accuracy'], step=0)
        
        self.writers['performance'].flush()
    
    def log_prediction_quality(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               fold: Optional[int] = None):
        """Log prediction quality metrics that traders care about"""
        # Ensure arrays are flattened for consistent calculations
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Handle potential length mismatch, though they should be the same
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            logger.warning(f"Mismatch in prediction and ground truth lengths. Trimmed to {min_len} samples.")

        # Calculate trader-relevant metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Avoid division by zero for MAPE and price_error_pct
        safe_y_true = np.where(y_true == 0, 1e-9, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100
        
        # Directional accuracy (most important for traders)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            correct_direction = (true_direction == pred_direction)

            # As per user request, introduce a tolerance. A prediction is also a "win" if the
            # predicted price is within 0.0005 of the actual price, even if direction is wrong.
            # This check is applied to the price at the end of each step in the diff.
            price_difference = np.abs(y_true[1:] - y_pred[1:])
            within_tolerance = (price_difference < 0.0005)
            
            # Combine the conditions: correct direction OR within price tolerance
            directional_accuracy = np.mean(correct_direction | within_tolerance) * 100
        else:
            directional_accuracy = 0
        
        # Price level accuracy
        price_error_pct = np.abs((y_true - y_pred) / safe_y_true) * 100
        
        step = fold if fold is not None else 0
        suffix = f'_Fold_{fold}' if fold is not None else '_Final'
        
        with self.writers['predictions'].as_default():
            tf.summary.scalar(f'Accuracy_{self.horizon}h/RMSE{suffix}', rmse, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/MAE{suffix}', mae, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/MAPE{suffix}', mape, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/Directional{suffix}', 
                            directional_accuracy, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/Avg_Price_Error_Pct{suffix}', 
                            np.mean(price_error_pct), step=step)
        
        self.writers['predictions'].flush()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def log_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Visualize prediction vs actual distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title(f'{self.horizon}h Prediction Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = ((y_pred - y_true) / y_true) * 100
        axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Prediction Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.close()
        
        with self.writers['predictions'].as_default():
            tf.summary.image(f'Prediction_Analysis_{self.horizon}h', image, step=0)
        
        self.writers['predictions'].flush()
    
    def log_backtest_chart(self, timestamps: pd.DatetimeIndex, 
                          y_true: np.ndarray, y_pred: np.ndarray):
        """Log backtest visualization"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        ax.plot(timestamps, y_true_flat, label='Actual Price', 
                color='#00FF00', linewidth=2, alpha=0.8)
        ax.plot(timestamps, y_pred_flat, label='Predicted Price', 
                color='#FF00FF', linewidth=2, linestyle='--', alpha=0.8)
        
        # Highlight prediction errors
        error_pct = np.abs((y_true_flat - y_pred_flat) / y_true_flat) * 100
        colors = ['green' if e < 2 else 'yellow' if e < 5 else 'red' 
                  for e in error_pct]
        ax.scatter(timestamps, y_pred_flat, c=colors, s=20, alpha=0.6, 
                  label='Error: Green<2%, Yellow<5%, Red>5%')
        
        ax.set_title(f'{self.horizon}h Backtest Performance', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.close()
        
        with self.writers['predictions'].as_default():
            tf.summary.image(f'Backtest_{self.horizon}h', image, step=0)
        
        self.writers['predictions'].flush()
    
    def log_feature_importance(self, feature_names: list, importances: np.ndarray):
        """Log feature importance for interpretability"""
        # Sort features by importance
        indices = np.argsort(importances)[-20:]  # Top 20
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_importances)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 20 Features for {self.horizon}h Prediction')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.close()
        
        with self.writers['training'].as_default():
            tf.summary.image(f'Feature_Importance_{self.horizon}h', image, step=0)
        
        self.writers['training'].flush()
    
    def close(self):
        """Close all writers"""
        for writer in self.writers.values():
            writer.close()

class KerasRegressorWrapper:
    """
    Wrapper for Keras models to make them scikit-learn compatible.
    """
    def __init__(self, build_fn, model_type: str, epochs: int = 100, **hyperparams):
        self.build_fn = build_fn
        self.model_type = model_type
        self.epochs = epochs
        self.hyperparams = hyperparams
        self.model = None
        self.log_dir = None

    def fit(self, X, y, validation_data=None, target_scaler=None, run_log_dir: Path = None):
        input_shape = (X.shape[1], X.shape[2])
        
        if self.model_type == 'cnn_lstm':
            self.model = self.build_fn(
                input_shape,
                conv_filters=self.hyperparams.get('conv_filters', 64),
                kernel_size=self.hyperparams.get('kernel_size', 3),
                n_units=self.hyperparams.get('n_units', 128),
                dropout=self.hyperparams.get('dropout', 0.2)
            )
        else:
            self.model = self.build_fn(
                input_shape,
                n_units=self.hyperparams.get('n_units', 128),
                dropout=self.hyperparams.get('dropout', 0.2),
                model_type=self.model_type
            )

        optimizer = Adam(learning_rate=self.hyperparams.get('learning_rate', 0.001))
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        if run_log_dir:
            self.log_dir = run_log_dir / self.model_type
            file_writer = tf.summary.create_file_writer(str(self.log_dir))
        else:
            file_writer = None

        callbacks = [StopIfNaN()]
        
        if validation_data is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=TRAINING_CONFIG["CV_PATIENCE"], 
                restore_best_weights=True, 
                min_delta=0.0001
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6, 
                verbose=0
            )
            callbacks.extend([early_stopping, reduce_lr])

        else:
            reduce_lr = ReduceLROnPlateau(
                monitor='loss',
                factor=0.5, 
                patience=5, 
                min_lr=1e-6, 
                verbose=0
            )
            callbacks.append(reduce_lr)

        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=32,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )

        if file_writer:
            with file_writer.as_default():
                tf.summary.scalar('final_loss', history.history['loss'][-1], step=self.epochs)
                if 'val_loss' in history.history:
                    tf.summary.scalar('final_val_loss', history.history['val_loss'][-1], step=self.epochs)

        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0)

# OPTIMIZED: Vectorized confidence interval calculation
def calculate_confidence_metrics(predictions: np.ndarray) -> dict:
    """
    Calculates improved confidence metrics from ensemble predictions using vectorized operations.
    """
    # Remove outliers using IQR (vectorized for all samples at once)
    q1 = np.percentile(predictions, 25, axis=1, keepdims=True)
    q3 = np.percentile(predictions, 75, axis=1, keepdims=True)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # Mask outliers
    mask = (predictions >= lower_bound) & (predictions <= upper_bound)
    
    # Calculate metrics using masked arrays
    metrics = {}
    mean_preds = []
    std_preds = []
    ci_vals = []
    conf_scores = []
    ci_lower_q = []
    ci_upper_q = []
    
    for i in range(predictions.shape[0]):
        clean_preds = predictions[i][mask[i]]
        
        if len(clean_preds) < 2:
            clean_preds = predictions[i]
        
        mean_pred = np.mean(clean_preds)
        std_pred = np.std(clean_preds)
        
        # Quantile-based
        ci_lower_q.append(np.percentile(clean_preds, 10))
        ci_upper_q.append(np.percentile(clean_preds, 90))
        
        # Adaptive CI
        base_factor = 1.645
        agreement_factor = np.clip(1 / (1 + (std_pred / (mean_pred + 1e-9)) * 10), 0.3, 1.0)
        adaptive_ci = base_factor * std_pred * agreement_factor
        
        # Confidence score
        normalized_std = min(std_pred / (mean_pred + 1e-9), 0.1)
        confidence = (1 - (normalized_std / 0.1)) * 100
        
        mean_preds.append(mean_pred)
        std_preds.append(std_pred)
        ci_vals.append(adaptive_ci)
        conf_scores.append(confidence)
    
    return {
        'mean': np.array(mean_preds),
        'std': np.array(std_preds),
        'ci': np.array(ci_vals),
        'confidence_score': np.array(conf_scores),
        'ci_lower_quantile': np.array(ci_lower_q),
        'ci_upper_quantile': np.array(ci_upper_q)
    }

# OPTIMIZED: Memory-efficient sequence creation using stride_tricks
def create_sequences_optimized(data, seq_len):
    """
    Creates sequences using numpy stride tricks for memory efficiency.
    """
    if len(data) <= seq_len:
        return np.array([])
    
    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples = len(data) - seq_len
    n_features = data.shape[1]
    
    # Calculate shape and strides for windowing
    shape = (n_samples, seq_len, n_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    
    # Create view using stride tricks (zero-copy)
    sequences = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    # Return a copy to avoid stride issues with subsequent operations
    return sequences.copy()



# --- 6. ENHANCED TRAINING & EVALUATION ---
def train_and_evaluate_for_horizon(horizon: int, run_log_dir: Path, sequence_length: int = TRAINING_CONFIG["SEQUENCE_LENGTH"], ticker: str = TRAINING_CONFIG["TICKER"], timeframe: str = TRAINING_CONFIG["TIMEFRAME"]) -> None:
    """
    Trains, evaluates, and saves all models for a specific prediction horizon using a FAST single train/test split.
    """
    TerminalStyle.header(f"TRAINING PIPELINE: {horizon}H PREDICTION HORIZON")
    
    tb_logger = TradingTensorBoard(run_log_dir, horizon)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path('sentiment').mkdir(parents=True, exist_ok=True)
    
    TerminalStyle.subheader("Phase 1: Data Acquisition")
    data = load_data(ticker, timeframe, limit=TRAINING_CONFIG["DATA_LIMIT"])
    data = fetch_and_add_sentiment(data, ticker.split('/')[0])
    
    whale_df = pd.DataFrame()
    try:
        data, whale_df = asyncio.run(integrate_onchain_metrics(data, ticker.split('/')[0]))
    except RuntimeError as e:
        if 'already running' in str(e):
            loop = asyncio.get_event_loop()
            data, whale_df = loop.run_until_complete(integrate_onchain_metrics(data, ticker.split('/')[0]))
        else:
            raise

    TerminalStyle.subheader("Phase 2: Feature Engineering")
    featured_data = feature_engineering(data)
    
    tb_logger.log_market_context(featured_data)
    if not whale_df.empty:
        tb_logger.log_whale_activity(whale_df)

    log_dir = run_log_dir / f'evaluation_{horizon}h'
    featured_data = detect_market_regime(featured_data, str(log_dir))
    
    featured_data['target'] = featured_data['close'].shift(-horizon)
    featured_data.dropna(inplace=True)
    
    y, X = featured_data[['target']], featured_data.drop(columns='target')
    
    # ============================================
    # FAST TRAIN/TEST SPLIT - NO WALK-FORWARD
    # ============================================
    TerminalStyle.subheader("Phase 3: Train/Test Split (80/20)")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    TerminalStyle.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Scale data
    feature_scaler = RobustScaler()
    target_scaler = MinMaxScaler()
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)
    X_test_scaled = feature_scaler.transform(X_test)

    feature_columns = X_train.columns.tolist()

    # Create sequences
    X_train_seq = create_sequences_optimized(X_train_scaled, sequence_length)
    y_train_seq = y_train_scaled[sequence_length:]
    X_test_seq = create_sequences_optimized(X_test_scaled, sequence_length)
    y_test_seq = y_test.values[sequence_length:]
    
    X_train_tab = X_train_scaled[sequence_length:]
    X_test_tab = X_test_scaled[sequence_length:]
    X_train_tab_df = pd.DataFrame(X_train_tab, columns=feature_columns)
    X_test_tab_df = pd.DataFrame(X_test_tab, columns=feature_columns)
    
    # Train base models ONCE
    model_names = ['gru', 'lstm', 'cnn_lstm', 'lgbm', 'xgb']
    base_models = {}
    
    for i, name in enumerate(model_names, 1):
        TerminalStyle.info(f"Training {name.upper()} ({i}/{len(model_names)})...")
        
        if name in ['gru', 'lstm', 'cnn_lstm']:
            epochs = TRAINING_CONFIG["CV_EPOCHS_DL"]
            hyperparams = get_default_hyperparameters(name)
            
            if name == 'gru':
                base_models[name] = KerasRegressorWrapper(
                    create_dl_model, model_type='gru', epochs=epochs, **hyperparams
                ).fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), target_scaler=target_scaler, run_log_dir=run_log_dir)
            elif name == 'lstm':
                base_models[name] = KerasRegressorWrapper(
                    create_dl_model, model_type='lstm', epochs=epochs, **hyperparams
                ).fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), target_scaler=target_scaler, run_log_dir=run_log_dir)
            elif name == 'cnn_lstm':
                base_models[name] = KerasRegressorWrapper(
                    create_cnn_lstm_model, model_type='cnn_lstm', epochs=epochs, **hyperparams
                ).fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), target_scaler=target_scaler, run_log_dir=run_log_dir)
        else:
            params = TRAINING_CONFIG["MODEL_PARAMS"].get(name, {})
            if name == 'lgbm':
                base_models[name] = LGBMRegressor(**params).fit(X_train_tab_df, y_train_seq.ravel())
            elif name == 'xgb':
                base_models[name] = xgb.XGBRegressor(**params).fit(X_train_tab_df, y_train_seq.ravel())
        
        TerminalStyle.success(f"{name.upper()} trained")

    # Train meta-model
    TerminalStyle.subheader("Phase 4: Meta-Model Training")
    
    meta_features_train = np.column_stack([
        base_models[name].predict(X_train_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_train_tab_df)
        for name in base_models
    ])
    
    meta_model = Ridge().fit(meta_features_train, y_train_seq.ravel())
    
    # Evaluate on test set
    meta_features_test = np.column_stack([
        base_models[name].predict(X_test_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_test_tab_df)
        for name in base_models
    ])
    
    final_predictions_scaled = meta_model.predict(meta_features_test).reshape(-1, 1)
    final_predictions = target_scaler.inverse_transform(final_predictions_scaled).flatten()
    
    # Log metrics
    metrics = tb_logger.log_prediction_quality(y_test_seq, final_predictions)
    
    TerminalStyle.subheader("Phase 5: Test Set Performance")
    TerminalStyle.metric("RMSE", f"{metrics['rmse']:.4f}")
    TerminalStyle.metric("MAE", f"{metrics['mae']:.4f}")
    TerminalStyle.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1f}%", 
                        positive=(metrics['directional_accuracy'] > 50))
    
    # Backtest visualization
    timestamps_test = y_test.index[sequence_length:]
    tb_logger.log_backtest_chart(timestamps_test, y_test_seq, final_predictions)
    tb_logger.log_prediction_distribution(y_test_seq, final_predictions)
    
    # Save models and scalers
    TerminalStyle.subheader("Phase 6: Saving Models")
    for name in base_models:
        joblib.dump(base_models[name], MODEL_DIR / f'base_{name}_{horizon}h.pkl')
    
    joblib.dump(meta_model, MODEL_DIR / f'meta_model_{horizon}h.pkl')
    joblib.dump(feature_columns, MODEL_DIR / f'feature_columns_{horizon}h.pkl')
    joblib.dump(feature_scaler, MODEL_DIR / f'features_scaler_{horizon}h.pkl')
    joblib.dump(target_scaler, MODEL_DIR / f'target_scaler_{horizon}h.pkl')
    
    TerminalStyle.success("All models and scalers saved")
    
    if hasattr(base_models.get('lgbm'), 'feature_importances_'):
        tb_logger.log_feature_importance(feature_columns, base_models['lgbm'].feature_importances_)
    
    tb_logger.close()



def verify_and_rotate_plots():
    """Handles the rotation and verification of historical forecast plots at the start of a run."""
    TerminalStyle.subheader("Phase 0: Verifying Previous Forecast")
    prediction_dir = Path('predictions')
    
    f1 = prediction_dir / 'future_forecast_1.png'
    f2 = prediction_dir / 'future_forecast_2.png'
    f3 = prediction_dir / 'future_forecast_3.png'

    # Rotate _2 -> _3 and _1 -> _2
    if f2.exists():
        TerminalStyle.info("Rotating future_forecast_2.png to future_forecast_3.png")
        if f3.exists():
            f3.unlink()
        f2.rename(f3)
    
    if f1.exists():
        TerminalStyle.info("Rotating future_forecast_1.png to future_forecast_2.png")
        f1.rename(f2)
    else:
        TerminalStyle.info("No previous forecast (future_forecast_1.png) to verify.")
        return

    # Now, f2 contains the plot from the previous run. We will update it with actuals.
    TerminalStyle.info("Verifying previous forecast and updating future_forecast_2.png...")
    
    try:
        history_df = pd.read_csv(prediction_dir / 'predictions.csv', parse_dates=['timestamp'])
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], utc=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        TerminalStyle.warning("Could not read prediction history. Skipping verification.")
        return

    if history_df.empty:
        TerminalStyle.warning("Prediction history is empty. Skipping verification.")
        return

    # Find the second to last prediction run from the CSV
    unique_timestamps = sorted(history_df['timestamp'].unique(), reverse=True)
    if len(unique_timestamps) < 2:
        TerminalStyle.info("Not enough historical runs to verify.")
        return

    last_run_timestamp = unique_timestamps[1]
    last_run_preds = history_df[history_df['timestamp'] == last_run_timestamp].copy()

    if last_run_preds.empty:
        TerminalStyle.warning(f"Could not find predictions for run at {last_run_timestamp}. Skipping verification.")
        return

    ticker = os.getenv("TICKER", "XRP/USDT")
    timeframe = os.getenv("TIMEFRAME", "1h")
    
    last_run_preds['horizon_hours'] = last_run_preds['horizon'].str.replace('h', '').astype(int)
    max_h = last_run_preds['horizon_hours'].max()
    
    start_time = pd.to_datetime(last_run_preds['timestamp'].iloc[0])
    end_time = pd.Timestamp.now(tz='UTC')
    hours_to_fetch = int((end_time - start_time).total_seconds() / 3600) + max_h + 2

    TerminalStyle.info(f"Fetching latest price data for verification window...")
    actuals_df = load_data(ticker, timeframe, limit=hours_to_fetch)
    verification_actuals = actuals_df[(actuals_df.index >= start_time) & (actuals_df.index <= end_time)]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['#8A2BE2', '#FF69B4', '#40E0D0', '#FFA500']

    start_price = last_run_preds['current_price'].iloc[0]

    for i, row in enumerate(last_run_preds.itertuples()):
        future_time = start_time + pd.Timedelta(hours=row.horizon_hours)
        color = colors[i % len(colors)]
        ax.plot([start_time, future_time], [start_price, row.predicted_price], linestyle='--', color=color, alpha=0.7, linewidth=2)
        ax.plot(future_time, row.predicted_price, 'x', color=color, markersize=12, markeredgewidth=3, label=f'{row.horizon} Prediction', zorder=5)

    ax.plot(start_time, start_price, 'o', color='white', markersize=12, zorder=11, label='Original Forecast Price')

    if not verification_actuals.empty:
        ax.plot(verification_actuals.index, verification_actuals['close'], color='white', linewidth=2.5, label='Actual Price Path', zorder=10, alpha=0.9)

    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'{ticker} Forecast vs Actual (Prediction from {start_time.strftime("%Y-%m-%d %H:%M")})', fontsize=16, color='white', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price (USDT)', fontsize=12, color='white')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555', alpha=0.3)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(f2, dpi=300, bbox_inches='tight')
    plt.close()
    TerminalStyle.success(f"Verification chart updated and saved to {f2.name}")

def plot_future_forecast(predictions_df, ticker):
    """
    Plots ALL future prediction horizons on a single chart with a dark theme.
    """
    if predictions_df.empty:
        return

    # Use the latest run by finding the most recent timestamp
    latest_timestamp = predictions_df['timestamp'].max()
    latest_preds = predictions_df[predictions_df['timestamp'] == latest_timestamp].copy()

    if latest_preds.empty:
        return

    # --- Chart Styling ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define the color palette - one color per horizon
    colors = ['#8A2BE2', '#FF69B4', '#40E0D0', '#FFA500']  # Purple, Pink, Turquoise, Orange

    # --- Data Preparation ---
    latest_preds['horizon_hours'] = latest_preds['horizon'].str.replace('h', '').astype(int)
    latest_preds = latest_preds.sort_values('horizon_hours')
    
    # Ensure we have all horizons before plotting
    if len(latest_preds) == 0:
        logger.warning("No predictions found for the latest timestamp")
        return
    
    start_time = pd.to_datetime(latest_preds['timestamp'].iloc[0])
    start_price = latest_preds['current_price'].iloc[0]

    # Plot current price as a starting point
    ax.plot(start_time, start_price, 'o', color='white', markersize=12, label='Current Price', zorder=10)

    # --- Plot ALL Horizons ---
    for i, row in enumerate(latest_preds.itertuples()):
        future_time = start_time + pd.Timedelta(hours=row.horizon_hours)
        color = colors[i % len(colors)]
        
        # Dashed line from current to predicted
        ax.plot([start_time, future_time], [start_price, row.predicted_price], 
                linestyle='--', color=color, alpha=0.8, linewidth=2)
        
        # Predicted price point
        ax.plot(future_time, row.predicted_price, 'o', color=color, markersize=10, 
                label=f'{row.horizon} Prediction', zorder=5)
        
        # Confidence interval shaded area
        # if hasattr(row, 'confidence_interval_lower') and hasattr(row, 'confidence_interval_upper'):
        #     # Create polygon for shaded confidence area
        #     times = [start_time, future_time, future_time, start_time]
        #     prices = [start_price, row.confidence_interval_upper, row.confidence_interval_lower, start_price]
        #     ax.fill(times, prices, color=color, alpha=0.15)

    # --- Formatting ---
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'{ticker} Future Price Forecast', fontsize=18, color='white', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price (USDT)', fontsize=12, color='white')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555', alpha=0.3)
    
    # Format x-axis to show dates nicely
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()

        # --- Saving ---
    prediction_dir = Path('predictions')
    base_name = "future_forecast"
    plot_path = prediction_dir / f'{base_name}_1.png'  # Always save new plot as _1
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    TerminalStyle.success(f"Future forecast plot saved to {plot_path}")
    
# --- 7. FUTURE PREDICTION WITH IMPROVED UNCERTAINTY ---
def make_future_predictions(horizons: List[int], sequence_length: int = TRAINING_CONFIG["SEQUENCE_LENGTH"], ticker: str = TRAINING_CONFIG["TICKER"], timeframe: str = TRAINING_CONFIG["TIMEFRAME"]) -> None:
    """
    Generates and saves future price predictions for multiple horizons with IMPROVED confidence intervals.
    """
    TerminalStyle.header("GENERATING FUTURE PREDICTIONS WITH IMPROVED CONFIDENCE INTERVALS")
    
    Path('sentiment').mkdir(parents=True, exist_ok=True)
    Path('predictions').mkdir(parents=True, exist_ok=True)

    # --- Model Caching ---
    TerminalStyle.subheader("Caching Models")
    model_cache = {}
    for horizon in horizons:
        try:
            model_cache[horizon] = {
                'base_models': {name: joblib.load(MODEL_DIR / f'base_{name}_{horizon}h.pkl') for name in ['gru', 'lstm', 'cnn_lstm', 'lgbm', 'xgb']},
                'meta_model': joblib.load(MODEL_DIR / f'meta_model_{horizon}h.pkl'),
                'feature_columns': joblib.load(MODEL_DIR / f'feature_columns_{horizon}h.pkl'),
                'features_scaler': joblib.load(MODEL_DIR / f'features_scaler_{horizon}h.pkl'),
                'target_scaler': joblib.load(MODEL_DIR / f'target_scaler_{horizon}h.pkl')
            }
            TerminalStyle.success(f"Models for {horizon}h cached")
        except FileNotFoundError as e:
            logger.error(f"Models for {horizon}h not found. Please train first")
            return
    
    TerminalStyle.success("All models cached successfully")

    TerminalStyle.subheader("Loading Latest Market Data")
    data = load_data(ticker, timeframe, limit=sequence_length + 200)
    data = fetch_and_add_sentiment(data, ticker.split('/')[0])
    try:
        data, _ = asyncio.run(integrate_onchain_metrics(data, ticker.split('/')[0]))
    except RuntimeError as e:
        if 'already running' in str(e):
            loop = asyncio.get_event_loop()
            data, _ = loop.run_until_complete(integrate_onchain_metrics(data, ticker.split('/')[0]))
        else:
            raise
    featured_data = feature_engineering(data)
    featured_data = detect_market_regime(featured_data)

    all_predictions = []
    current_price = featured_data['close'].iloc[-1]
    
    print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}Current {ticker} Price: {TerminalColors.PINK}{current_price:.4f} USDT{TerminalColors.ENDC}\n")

    timestamp = pd.Timestamp.now(tz='UTC')
    for horizon in horizons:
        try:
            if horizon not in model_cache:
                logger.warning(f"Models for {horizon}h not in cache. Skipping")
                continue

            cached_models = model_cache[horizon]
            base_models = cached_models['base_models']
            meta_model = cached_models['meta_model']
            feature_columns = cached_models['feature_columns']
            feature_scaler = cached_models['features_scaler']
            target_scaler = cached_models['target_scaler']

            last_sequence_unscaled = featured_data.tail(sequence_length).drop(columns=['target'], errors='ignore')
            last_sequence_scaled = feature_scaler.transform(last_sequence_unscaled)
            X_future_seq = np.array([last_sequence_scaled])
            X_future_tab_df = pd.DataFrame([last_sequence_scaled[-1]], columns=feature_columns)
            
            # Get predictions from all base models
            base_predictions_scaled = np.column_stack([
                base_models[name].predict(X_future_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_future_tab_df)
                for name in base_models
            ])
            
            # Calculate uncertainty using IMPROVED adaptive intervals
            uncertainty_metrics = calculate_confidence_metrics(base_predictions_scaled)
            
            # Meta-model prediction
            future_price_scaled = meta_model.predict(base_predictions_scaled.reshape(1, -1))
            future_price = target_scaler.inverse_transform(future_price_scaled.reshape(-1, 1)).flatten()[0]
            
            # Transform ADAPTIVE confidence intervals
            ci_lower_scaled = future_price_scaled[0] - uncertainty_metrics['ci'][0]
            ci_upper_scaled = future_price_scaled[0] + uncertainty_metrics['ci'][0]
            ci_lower = target_scaler.inverse_transform([[ci_lower_scaled]])[0][0]
            ci_upper = target_scaler.inverse_transform([[ci_upper_scaled]])[0][0]
            
            # Also get quantile-based intervals for comparison
            quantile_lower = target_scaler.inverse_transform([[uncertainty_metrics['ci_lower_quantile'][0]]])[0][0]
            quantile_upper = target_scaler.inverse_transform([[uncertainty_metrics['ci_upper_quantile'][0]]])[0][0]
            
            confidence_score = uncertainty_metrics['confidence_score'][0]
            ci_width_pct = ((ci_upper - ci_lower) / future_price) * 100
            
            price_change = ((future_price - current_price) / current_price) * 100
            
            # Display with confidence
            TerminalStyle.prediction_box(
                current_price, 
                future_price, 
                price_change, 
                horizon,
                ci_lower,
                ci_upper,
                confidence_score
            )
            
            # Confidence interpretation with CI width info
            if confidence_score >= 75:
                TerminalStyle.success(f"High confidence prediction - Low market uncertainty detected")
                TerminalStyle.info(f"✓ Adaptive CI narrowed to ±{ci_width_pct:.1f}% (vs Quantile: ±{((quantile_upper-quantile_lower)/future_price)*100:.1f}%)")
            elif confidence_score >= 50:
                TerminalStyle.warning(f"Medium confidence - Moderate market uncertainty")
                TerminalStyle.info(f"CI width: ±{ci_width_pct:.1f}% of predicted price")
            else:
                TerminalStyle.warning(f"Low confidence - High market uncertainty detected. Use with caution!")
                TerminalStyle.info(f"Wide CI: ±{ci_width_pct:.1f}% due to model disagreement")
            
            TerminalStyle.info(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            all_predictions.append({
                'horizon': f'{horizon}h',
                'current_price': current_price,
                'predicted_price': future_price,
                'expected_change_pct': price_change,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'prediction_std': uncertainty_metrics['std'][0],
                'confidence_score': confidence_score,
                'ci_width_pct': ci_width_pct,
                'timestamp': timestamp
            })
            
        except Exception as e:
            logger.error(f"Prediction error for {horizon}h: {e}")
    
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        log_path = Path('predictions') / 'predictions.csv'
        pred_df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        
        print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * 80}{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{'PREDICTION SUMMARY WITH IMPROVED CONFIDENCE'.center(78)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
        print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * 80}{TerminalColors.ENDC}\n")
        
        for pred in all_predictions:
            color = TerminalColors.OKGREEN if pred['expected_change_pct'] > 0 else TerminalColors.FAIL
            symbol = "▲" if pred['expected_change_pct'] > 0 else "▼"
            conf_color = TerminalColors.OKGREEN if pred['confidence_score'] >= 75 else TerminalColors.ORANGE if pred['confidence_score'] >= 50 else TerminalColors.FAIL
            
            print(f"  {TerminalColors.BOLD}{pred['horizon']:>4}{TerminalColors.ENDC} │ "
                  f"Predicted: {TerminalColors.BOLD}{pred['predicted_price']:.4f}{TerminalColors.ENDC} │ "
                  f"Range: {TerminalColors.DIM}{pred['confidence_interval_lower']:.4f}-{pred['confidence_interval_upper']:.4f}{TerminalColors.ENDC} │ "
                  f"Change: {color}{symbol} {abs(pred['expected_change_pct']):.2f}%{TerminalColors.ENDC} │ "
                  f"Confidence: {conf_color}{pred['confidence_score']:.0f}%{TerminalColors.ENDC} │ "
                  f"CI: {TerminalColors.DIM}±{pred['ci_width_pct']:.1f}%{TerminalColors.ENDC}")
        
        print(f"\n{TerminalColors.DIM}Predictions saved to: {log_path}{TerminalColors.ENDC}")
        TerminalStyle.info(f"Using {CI_METHOD.upper()} confidence interval method")

        # Generate and save the plot - THIS SHOULD ONLY BE CALLED ONCE
        plot_future_forecast(pred_df, ticker)
    else:
        TerminalStyle.warning("No predictions were generated")



# --- 8. MAIN EXECUTION ---
if __name__ == '__main__':
    import shutil
    from datetime import datetime

    # Enhanced startup banner
    width = 80
    print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * width}{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{'CLAIRVOYANT v3.2 - XRP PRICE FORECASTER'.center(width-2)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PURPLE}{'Multi-Model. Market Analysis. News Sentiment. On-Chain Whale Metrics'.center(width-2)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.OKCYAN}{'— created by Kevin ₿ourn —'.center(width-2)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{'- —⋆🔮︎⋆— -'.center(width-2)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * width}{TerminalColors.ENDC}\n")
   
    TerminalStyle.info(f"Confidence Interval Method: {CI_METHOD.upper()}")
    TerminalStyle.info(f"Prediction Horizons: {', '.join([f'{h}h' for h in TRAINING_CONFIG['PREDICTION_HORIZONS']])}")
    TerminalStyle.info(f"Hyperparameter Optimization: {'ENABLED' if TRAINING_CONFIG['OPTIMIZE_HYPERPARAMETERS'] else 'DISABLED'}")
    TerminalStyle.info("On-Chain Metrics: ENABLED (XRP Ledger)")
    
    # --- Log Rotation ---
    TerminalStyle.subheader("Managing Log Directories")
    log_root = Path('logs')
    log_root.mkdir(exist_ok=True)
    
    run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    runs_to_keep = 2
    if len(run_dirs) > runs_to_keep:
        runs_to_delete = run_dirs[:-runs_to_keep]
        TerminalStyle.info(f"Found {len(run_dirs)} old log runs. Deleting {len(runs_to_delete)} to keep the last {runs_to_keep}.")
        for old_run_dir in runs_to_delete:
            try:
                shutil.rmtree(old_run_dir)
                TerminalStyle.success(f"Removed old log directory: {old_run_dir}")
            except OSError as e:
                TerminalStyle.error(f"Error removing directory {old_run_dir}: {e}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_log_dir = log_root / f"run_{run_timestamp}"
    current_run_log_dir.mkdir()
    TerminalStyle.success(f"Created new log directory for this run: {current_run_log_dir}")

    # --- Main Execution Logic ---
    verify_and_rotate_plots()

    parser = argparse.ArgumentParser(description='Clairvoyant v3.2 - XRP Price Forecaster')
    parser.add_argument('--train', action='store_true', help='Run the training pipeline.')
    parser.add_argument('--predict', action='store_true', help='Run the prediction pipeline.')
    args = parser.parse_args()

    # If no flags are specified, run both training and prediction
    if not args.train and not args.predict:
        args.train = True
        args.predict = True

    if args.train:
        # Train models for each horizon
        for h in TRAINING_CONFIG["PREDICTION_HORIZONS"]:
            train_and_evaluate_for_horizon(horizon=h, run_log_dir=current_run_log_dir)
    
    if args.predict:
        # Generate future predictions and the forecast plot
        make_future_predictions(horizons=TRAINING_CONFIG["PREDICTION_HORIZONS"])

    print(f"\n{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * 80}{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}║{TerminalColors.PINK}{'CLAIRVOYANT RUN COMPLETE'.center(78)}{TerminalColors.TURQUOISE}║{TerminalColors.ENDC}")
    print(f"{TerminalColors.BOLD}{TerminalColors.TURQUOISE}{'═' * 80}{TerminalColors.ENDC}\n")
    
    TerminalStyle.info("All requested operations completed successfully.")
    TerminalStyle.info(f"Models are saved in: {MODEL_DIR}")
    TerminalStyle.info(f"Predictions and charts are in: {Path('predictions')}")
    TerminalStyle.info(f"Check TensorBoard logs for detailed training metrics: tensorboard --logdir={log_root}")
    
    print()