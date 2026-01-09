# ============================================================================
# CHUNK 1: IMPORTS AND CONFIGURATION (UPDATED - Option 1)
# Description: All imports, SSL bypass, API configuration, and constants
# Changes Made:
#   - REMOVED: Hardcoded API_KEY and API_SECRET
#   - Credentials will now be entered by user in login form
# Dependencies: None (this is the base chunk)
# ============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, date
import json
import ssl
import urllib3
import warnings
import os
from kiteconnect import KiteConnect, KiteTicker
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from threading import Thread, Lock
import queue
from collections import defaultdict

# --- SSL Bypass (Required for corporate networks) ---
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# FIX: Check if already patched before patching
if not hasattr(requests.Session, '_ssl_patched'):
    _original_request = requests.Session.request
    
    def patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return _original_request(self, method, url, **kwargs)
    
    requests.Session.request = patched_request
    requests.Session._ssl_patched = True  # Mark as patched

# --- API Configuration ---
# REMOVED: Hardcoded credentials
# API_KEY = "xxxxxxxxxxxx"      # <-- REMOVED
# API_SECRET = "xxxxxxxxxxxx"   # <-- REMOVED
# 
# Credentials are now entered by user in login form and stored in:
#   - st.session_state.api_key
#   - st.session_state.api_secret

# --- Market Timing Constants ---
MARKET_OPEN = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
MARKET_CLOSE = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
STOP_TRADING = datetime.now().replace(hour=15, minute=15, second=0, microsecond=0)

# --- Index Lot Sizes (Fixed) ---
INDEX_LOT_SIZES = {
    'NIFTY': 25,
    'BANKNIFTY': 15,
    'FINNIFTY': 25,
    'MIDCPNIFTY': 50
}

# --- Index Tokens (Fixed) ---
INDEX_TOKENS = {
    'NIFTY': 256265,
    'BANKNIFTY': 260105,
    'FINNIFTY': 257801,
    'MIDCPNIFTY': 288009
}

# ============================================================================
# END OF CHUNK 1 (UPDATED)
# Changes Summary:
# 1. Removed hardcoded API_KEY
# 2. Removed hardcoded API_SECRET
# 3. Added comments explaining credentials are now dynamic via session state
# Next: Chunk 2 - Session State Initialization
# ============================================================================

# ============================================================================
# CHUNK 2: SESSION STATE INITIALIZATION (UPDATED - Option 1)
# Description: Initialize all Streamlit session state variables
# Changes Made:
#   - ADDED: api_key - User-entered API key
#   - ADDED: api_secret - User-entered API secret  
#   - ADDED: credentials_saved - Flag to track if credentials are entered
#   - ADDED: clear_api_credentials() function
#   - ADDED: validate_api_credentials() function
# Dependencies: Chunk 1 (imports)
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables for the trading bot"""
    
    # --- API Credentials (NEW - Option 1) ---
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'api_secret' not in st.session_state:
        st.session_state.api_secret = None
    if 'credentials_saved' not in st.session_state:
        st.session_state.credentials_saved = False
    
    # --- Kite Connection ---
    if 'kite' not in st.session_state:
        st.session_state.kite = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    
    # --- WebSocket Manager ---
    if 'ws_manager' not in st.session_state:
        st.session_state.ws_manager = None
    if 'live_ticks' not in st.session_state:
        st.session_state.live_ticks = {}
    if 'tick_lock' not in st.session_state:
        st.session_state.tick_lock = Lock()
    
    # --- Trading Controls ---
    if 'kill_switch' not in st.session_state:
        st.session_state.kill_switch = False
    if 'paper_mode' not in st.session_state:
        st.session_state.paper_mode = True
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = False
    if 'max_concurrent_trades' not in st.session_state:
        st.session_state.max_concurrent_trades = 2
    
    # --- Capital Management ---
    if 'paper_capital' not in st.session_state:
        st.session_state.paper_capital = 100000
    if 'initial_paper_capital' not in st.session_state:
        st.session_state.initial_paper_capital = 100000
    
    # --- Daily Profit/Loss Limits ---
    if 'daily_profit_target' not in st.session_state:
        st.session_state.daily_profit_target = 5000
    if 'daily_stop_loss' not in st.session_state:
        st.session_state.daily_stop_loss = 3000
    if 'daily_target_reached' not in st.session_state:
        st.session_state.daily_target_reached = False
    if 'daily_sl_reached' not in st.session_state:
        st.session_state.daily_sl_reached = False
    if 'session_start_capital' not in st.session_state:
        st.session_state.session_start_capital = st.session_state.paper_capital
    if 'daily_realized_pnl' not in st.session_state:
        st.session_state.daily_realized_pnl = 0
    if 'trading_stopped_reason' not in st.session_state:
        st.session_state.trading_stopped_reason = None
    
    # --- Positions ---
    if 'paper_positions' not in st.session_state:
        st.session_state.paper_positions = []
    if 'live_positions' not in st.session_state:
        st.session_state.live_positions = []
    
    # --- Trade Logs ---
    if 'trade_logs' not in st.session_state:
        st.session_state.trade_logs = []
    
    # --- Stock Data ---
    if 'option_stocks' not in st.session_state:
        st.session_state.option_stocks = []
    if 'all_instruments' not in st.session_state:
        st.session_state.all_instruments = None
    if 'nfo_instruments' not in st.session_state:
        st.session_state.nfo_instruments = None
    
    # --- Signals ---
    if 'active_signals' not in st.session_state:
        st.session_state.active_signals = []
    if 'scanned_stocks' not in st.session_state:
        st.session_state.scanned_stocks = []
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None
    
    # --- Real-time Option Prices Cache ---
    if 'option_price_cache' not in st.session_state:
        st.session_state.option_price_cache = {}
    
    # --- Performance Stats ---
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
    
    # --- After Market Analysis ---
    if 'after_market_results' not in st.session_state:
        st.session_state.after_market_results = None
    
    # --- Tick Data Buffer for Candle Building ---
    if 'tick_buffer' not in st.session_state:
        st.session_state.tick_buffer = defaultdict(list)
    if 'realtime_candles' not in st.session_state:
        st.session_state.realtime_candles = {}
    
    # --- Index Only Mode ---
    if 'index_only_mode' not in st.session_state:
        st.session_state.index_only_mode = False


def calculate_daily_pnl():
    """Calculate total daily P&L including both realized and unrealized."""
    positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
    
    today = datetime.now().date()
    realized_pnl = 0
    for p in positions:
        if p['status'] == 'CLOSED':
            exit_time = p.get('exit_time')
            if exit_time and exit_time.date() == today:
                realized_pnl += p['pnl']
    
    unrealized_pnl = 0
    for p in positions:
        if p['status'] == 'OPEN':
            unrealized_pnl += p['pnl']
    
    total_pnl = realized_pnl + unrealized_pnl
    
    return {
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'total_pnl': total_pnl
    }


def check_daily_limits():
    """Check if daily profit target or stop loss has been reached."""
    pnl_info = calculate_daily_pnl()
    total_pnl = pnl_info['total_pnl']
    
    profit_target = st.session_state.daily_profit_target
    stop_loss = st.session_state.daily_stop_loss
    
    if profit_target > 0 and total_pnl >= profit_target:
        return True, 'PROFIT_TARGET', pnl_info
    
    if stop_loss > 0 and total_pnl <= -stop_loss:
        return True, 'DAILY_STOP_LOSS', pnl_info
    
    return False, None, pnl_info


def reset_daily_limits():
    """Reset daily limit flags - call this at start of new trading day."""
    st.session_state.daily_target_reached = False
    st.session_state.daily_sl_reached = False
    st.session_state.daily_realized_pnl = 0
    st.session_state.trading_stopped_reason = None
    st.session_state.session_start_capital = st.session_state.paper_capital


# ============================================================================
# NEW FUNCTIONS FOR OPTION 1
# ============================================================================

def clear_api_credentials():
    """
    Clear stored API credentials and reset connection state.
    Useful when user wants to login with different credentials.
    """
    st.session_state.api_key = None
    st.session_state.api_secret = None
    st.session_state.credentials_saved = False
    st.session_state.kite = None
    st.session_state.access_token = None
    
    # Disconnect WebSocket if connected
    if st.session_state.ws_manager:
        try:
            st.session_state.ws_manager.disconnect()
        except:
            pass
        st.session_state.ws_manager = None


def validate_api_credentials(api_key, api_secret):
    """
    Validate API credentials format.
    
    Args:
        api_key: Zerodha API key
        api_secret: Zerodha API secret
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if not api_key or not api_key.strip():
        return False, "API Key cannot be empty"
    
    if not api_secret or not api_secret.strip():
        return False, "API Secret cannot be empty"
    
    api_key = api_key.strip()
    api_secret = api_secret.strip()
    
    if len(api_key) < 8:
        return False, "API Key seems too short. Please check your credentials."
    
    if len(api_secret) < 20:
        return False, "API Secret seems too short. Please check your credentials."
    
    return True, None

# ============================================================================
# END OF CHUNK 2 (UPDATED)
# Changes Summary:
# 1. Added api_key session state variable
# 2. Added api_secret session state variable
# 3. Added credentials_saved flag
# 4. Added clear_api_credentials() function
# 5. Added validate_api_credentials() function
# Next: Chunk 3 - WebSocket Manager Class (NO CHANGES NEEDED)
# ============================================================================




# ============================================================================
# CHUNK 3: WEBSOCKET MANAGER CLASS
# Description: Handles real-time tick streaming from Zerodha KiteTicker
# Features:
#   - Connect/disconnect WebSocket
#   - Subscribe/unsubscribe tokens
#   - Store live ticks in shared state
#   - Build real-time candles from ticks
# Dependencies: Chunk 1 (imports), Chunk 2 (session state)
# ============================================================================

class WebSocketManager:
    """
    Manages real-time data streaming via Zerodha KiteTicker WebSocket.
    
    Usage:
        ws_manager = WebSocketManager(api_key, access_token)
        ws_manager.connect()
        ws_manager.subscribe_tokens([256265, 260105], mode='ltp')
        
        # Get live price
        price = ws_manager.get_ltp(256265)
    """
    
    def __init__(self, api_key, access_token):
        self.api_key = api_key
        self.access_token = access_token
        self.kws = None
        self.is_connected = False
        
        # Thread-safe data storage
        self.tick_lock = Lock()
        self.live_ticks = {}  # {token: tick_data}
        self.subscribed_tokens = set()
        
        # Tick buffer for candle building
        self.tick_buffer = defaultdict(list)
        self.candle_data = {}  # {token: DataFrame}
        
        # Callbacks
        self.on_tick_callback = None
        
    def connect(self):
        """Initialize and connect WebSocket"""
        try:
            self.kws = KiteTicker(self.api_key, self.access_token)
            
            # Assign callbacks
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            self.kws.on_reconnect = self._on_reconnect
            
            # Start WebSocket in a separate thread (non-blocking)
            self.kws.connect(threaded=True)
            
            # Wait for connection
            timeout = 10
            start = time.time()
            while not self.is_connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            return self.is_connected
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            if self.kws:
                self.kws.close()
                self.is_connected = False
        except:
            pass
    
    def _on_connect(self, ws, response):
        """Callback when WebSocket connects"""
        self.is_connected = True
        print(f"WebSocket Connected: {response}")
        
        # Re-subscribe to tokens if any
        if self.subscribed_tokens:
            self.kws.subscribe(list(self.subscribed_tokens))
            self.kws.set_mode(self.kws.MODE_FULL, list(self.subscribed_tokens))
    
    def _on_close(self, ws, code, reason):
        """Callback when WebSocket closes"""
        self.is_connected = False
        print(f"WebSocket Closed: {code} - {reason}")
    
    def _on_error(self, ws, code, reason):
        """Callback on WebSocket error"""
        print(f"WebSocket Error: {code} - {reason}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Callback on reconnection attempt"""
        print(f"WebSocket Reconnecting... Attempt {attempts_count}")
    
    def _on_ticks(self, ws, ticks):
        """
        Callback when ticks are received.
        Stores tick data in thread-safe dictionary.
        
        Tick structure (MODE_FULL):
        {
            'instrument_token': 256265,
            'last_price': 19500.50,
            'volume_traded': 1234567,
            'average_traded_price': 19480.25,
            'oi': 9876543,
            'oi_day_high': 10000000,
            'oi_day_low': 9500000,
            'ohlc': {'open': 19450, 'high': 19550, 'low': 19400, 'close': 19480},
            'depth': {'buy': [...], 'sell': [...]},
            'tradable': True,
            'last_traded_quantity': 75,
            'change': 0.52,
            'last_trade_time': datetime,
            'exchange_timestamp': datetime
        }
        """
        with self.tick_lock:
            for tick in ticks:
                token = tick['instrument_token']
                
                # Store latest tick
                self.live_ticks[token] = {
                    'ltp': tick.get('last_price', 0),
                    'volume': tick.get('volume_traded', 0),
                    'oi': tick.get('oi', 0),
                    'ohlc': tick.get('ohlc', {}),
                    'depth': tick.get('depth', {}),
                    'change': tick.get('change', 0),
                    'last_trade_time': tick.get('last_trade_time'),
                    'timestamp': datetime.now(),
                    'bid': tick.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                    'ask': tick.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                    'bid_qty': tick.get('depth', {}).get('buy', [{}])[0].get('quantity', 0),
                    'ask_qty': tick.get('depth', {}).get('sell', [{}])[0].get('quantity', 0),
                }
                
                # Add to buffer for candle building
                self.tick_buffer[token].append({
                    'price': tick.get('last_price', 0),
                    'volume': tick.get('volume_traded', 0),
                    'timestamp': datetime.now()
                })
                
                # Keep buffer size manageable (last 1000 ticks per token)
                if len(self.tick_buffer[token]) > 1000:
                    self.tick_buffer[token] = self.tick_buffer[token][-500:]
        
        # Call external callback if set
        if self.on_tick_callback:
            self.on_tick_callback(ticks)
    
    def subscribe_tokens(self, tokens, mode='full'):
        """
        Subscribe to instrument tokens for live data.
        
        Args:
            tokens: List of instrument tokens
            mode: 'ltp' (last price only), 'quote' (with depth), 'full' (everything)
        """
        if not self.kws or not self.is_connected:
            print("WebSocket not connected")
            return False
        
        try:
            # Add to subscribed set
            self.subscribed_tokens.update(tokens)
            
            # Subscribe
            self.kws.subscribe(tokens)
            
            # Set mode
            if mode == 'ltp':
                self.kws.set_mode(self.kws.MODE_LTP, tokens)
            elif mode == 'quote':
                self.kws.set_mode(self.kws.MODE_QUOTE, tokens)
            else:
                self.kws.set_mode(self.kws.MODE_FULL, tokens)
            
            print(f"Subscribed to {len(tokens)} tokens in {mode} mode")
            return True
            
        except Exception as e:
            print(f"Subscribe error: {e}")
            return False
    
    def unsubscribe_tokens(self, tokens):
        """Unsubscribe from tokens"""
        if not self.kws:
            return
        
        try:
            self.kws.unsubscribe(tokens)
            self.subscribed_tokens -= set(tokens)
            
            # Clean up tick data
            with self.tick_lock:
                for token in tokens:
                    self.live_ticks.pop(token, None)
                    self.tick_buffer.pop(token, None)
                    
        except Exception as e:
            print(f"Unsubscribe error: {e}")
    
    def get_ltp(self, token):
        """Get last traded price for a token"""
        with self.tick_lock:
            tick = self.live_ticks.get(token, {})
            return tick.get('ltp', 0)
    
    def get_tick(self, token):
        """Get full tick data for a token"""
        with self.tick_lock:
            return self.live_ticks.get(token, {}).copy()
    
    def get_all_ticks(self):
        """Get all live ticks"""
        with self.tick_lock:
            return dict(self.live_ticks)
    
    def get_bid_ask(self, token):
        """Get bid/ask prices for a token"""
        with self.tick_lock:
            tick = self.live_ticks.get(token, {})
            return {
                'bid': tick.get('bid', 0),
                'ask': tick.get('ask', 0),
                'bid_qty': tick.get('bid_qty', 0),
                'ask_qty': tick.get('ask_qty', 0)
            }
    
    def build_candles(self, token, interval_minutes=5):
        """
        Build OHLCV candles from tick buffer.
        
        Args:
            token: Instrument token
            interval_minutes: Candle interval in minutes
            
        Returns:
            DataFrame with OHLCV data
        """
        with self.tick_lock:
            ticks = self.tick_buffer.get(token, [])
        
        if not ticks:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(ticks)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to candles
        candles = df['price'].resample(f'{interval_minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        candles['volume'] = df['volume'].resample(f'{interval_minutes}T').last().diff().fillna(0)
        candles = candles.dropna()
        
        return candles.reset_index()
    
    def is_token_subscribed(self, token):
        """Check if token is subscribed"""
        return token in self.subscribed_tokens
    
    def get_subscribed_count(self):
        """Get count of subscribed tokens"""
        return len(self.subscribed_tokens)

# ============================================================================
# END OF CHUNK 3
# Next: Chunk 4 - Instrument and Option Contract Functions
# ============================================================================





# ============================================================================
# CHUNK 4: INSTRUMENT AND OPTION CONTRACT FUNCTIONS (UPDATED)
# Description: Functions to fetch instruments, option contracts, and real prices
# Changes Made:
#   - ADDED: is_expiry_day() function to check if today is expiry
#   - ADDED: get_days_to_expiry() function
#   - ADDED: should_avoid_expiry() filter function
#   - Modified get_atm_option_contract() to include expiry info
# Dependencies: Chunk 1-3
# ============================================================================

def is_expiry_day(expiry_date):
    """
    Check if today is the expiry day for given expiry date.
    
    IMPORTANT: Expiry day options are EXTREMELY risky due to:
    - Accelerated theta decay (time value drops to 0)
    - Extreme gamma (small moves cause huge premium swings)
    - Wide bid-ask spreads
    - Low liquidity in final hours
    
    Args:
        expiry_date: date object of option expiry
        
    Returns:
        bool: True if today is expiry day
    """
    today = datetime.now().date()
    
    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()
    
    return today == expiry_date


def get_days_to_expiry(expiry_date):
    """
    Calculate days remaining to expiry.
    
    Args:
        expiry_date: date object of option expiry
        
    Returns:
        int: Number of days to expiry (0 = expiry day)
    """
    today = datetime.now().date()
    
    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()
    
    delta = expiry_date - today
    return max(0, delta.days)


def should_avoid_expiry(expiry_date, min_days_to_expiry=1):
    """
    Determine if we should avoid trading this option due to expiry proximity.
    
    RULES:
    - ALWAYS avoid expiry day (0 DTE) - pure gambling
    - Optionally avoid 1 DTE for safety
    - For weekly options, be extra cautious
    
    Args:
        expiry_date: date object of option expiry
        min_days_to_expiry: Minimum days required (default 1 = avoid expiry day only)
        
    Returns:
        tuple: (should_avoid: bool, reason: str, days_to_expiry: int)
    """
    days_to_expiry = get_days_to_expiry(expiry_date)
    
    if days_to_expiry == 0:
        return True, "EXPIRY DAY - Extreme theta decay and gamma risk", 0
    
    if days_to_expiry < min_days_to_expiry:
        return True, f"Too close to expiry ({days_to_expiry} days < {min_days_to_expiry} required)", days_to_expiry
    
    # Warning for 1-2 DTE but allow
    if days_to_expiry <= 2:
        return False, f"WARNING: {days_to_expiry} DTE - Consider wider stops", days_to_expiry
    
    return False, f"OK - {days_to_expiry} days to expiry", days_to_expiry


def get_next_expiry_avoiding_current(kite, symbol, option_type, spot_price):
    """
    Get the NEXT expiry option contract, skipping current week if it's expiry day.
    
    This ensures we never trade expiry day options.
    
    Args:
        kite: KiteConnect instance
        symbol: Stock/Index symbol
        option_type: 'CE' or 'PE'
        spot_price: Current spot price
        
    Returns:
        dict: Option contract details for safe expiry
        None: If no suitable contract found
    """
    try:
        if st.session_state.nfo_instruments:
            instruments = st.session_state.nfo_instruments
        else:
            instruments = kite.instruments("NFO")
            st.session_state.nfo_instruments = instruments
        
        # Filter for this symbol and option type
        option_contracts = [
            inst for inst in instruments
            if symbol in inst['tradingsymbol']
            and inst['instrument_type'] == option_type
            and inst['expiry'] > datetime.now().date()
        ]
        
        if not option_contracts:
            return None
        
        # Sort by expiry
        option_contracts.sort(key=lambda x: x['expiry'])
        
        # Find first expiry that's NOT today
        safe_expiry = None
        for contract in option_contracts:
            avoid, reason, dte = should_avoid_expiry(contract['expiry'])
            if not avoid or dte >= 1:  # Allow if at least 1 day to expiry
                safe_expiry = contract['expiry']
                break
        
        if safe_expiry is None:
            return None
        
        # Filter for safe expiry
        safe_contracts = [c for c in option_contracts if c['expiry'] == safe_expiry]
        
        # Find ATM strike
        strikes = [c['strike'] for c in safe_contracts]
        if not strikes:
            return None
        
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # Get the ATM contract
        for contract in safe_contracts:
            if contract['strike'] == atm_strike:
                days_to_exp = get_days_to_expiry(contract['expiry'])
                return {
                    'tradingsymbol': contract['tradingsymbol'],
                    'token': contract['instrument_token'],
                    'strike': contract['strike'],
                    'expiry': contract['expiry'],
                    'lot_size': contract['lot_size'],
                    'instrument_type': contract['instrument_type'],
                    'exchange': 'NFO',
                    'days_to_expiry': days_to_exp,
                    'is_expiry_day': days_to_exp == 0
                }
        
        return None
        
    except Exception as e:
        print(f"Error getting safe expiry contract: {e}")
        return None


def get_all_option_stocks(kite):
    """
    Fetch all stocks available for options trading with ACTUAL lot sizes.
    
    Returns:
        List of dicts: [{symbol, token, exchange, lot_size}, ...]
    """
    try:
        # Fetch NFO instruments
        nfo_instruments = kite.instruments("NFO")
        st.session_state.nfo_instruments = nfo_instruments  # Cache for later use
        
        # Get unique underlying symbols with their lot sizes
        option_data = {}  # {symbol: lot_size}
        
        for inst in nfo_instruments:
            if inst['instrument_type'] in ['CE', 'PE']:
                tradingsymbol = inst['tradingsymbol']
                
                import re
                match = re.match(r'^([A-Z]+)', tradingsymbol)
                if match:
                    symbol = match.group(1)
                    
                    if len(symbol) < 2:
                        continue
                    
                    if symbol not in option_data:
                        option_data[symbol] = inst['lot_size']
        
        # Fetch NSE instruments to get tokens
        nse_instruments = kite.instruments("NSE")
        st.session_state.all_instruments = nse_instruments
        
        # Build final list
        option_stocks = []
        
        for symbol, lot_size in option_data.items():
            for nse_inst in nse_instruments:
                if nse_inst['tradingsymbol'] == symbol and nse_inst['instrument_type'] == 'EQ':
                    option_stocks.append({
                        'symbol': symbol,
                        'token': nse_inst['instrument_token'],
                        'exchange': 'NSE',
                        'lot_size': lot_size
                    })
                    break
        
        # Add indices
        indices = [
            {'symbol': 'NIFTY', 'token': INDEX_TOKENS['NIFTY'], 'exchange': 'NSE', 'lot_size': INDEX_LOT_SIZES['NIFTY']},
            {'symbol': 'BANKNIFTY', 'token': INDEX_TOKENS['BANKNIFTY'], 'exchange': 'NSE', 'lot_size': INDEX_LOT_SIZES['BANKNIFTY']},
            {'symbol': 'FINNIFTY', 'token': INDEX_TOKENS['FINNIFTY'], 'exchange': 'NSE', 'lot_size': INDEX_LOT_SIZES['FINNIFTY']},
            {'symbol': 'MIDCPNIFTY', 'token': INDEX_TOKENS['MIDCPNIFTY'], 'exchange': 'NSE', 'lot_size': INDEX_LOT_SIZES['MIDCPNIFTY']}
        ]
        
        option_stocks.extend(indices)
        
        return option_stocks
        
    except Exception as e:
        st.error(f"Error fetching option stocks: {e}")
        return []


def get_atm_option_contract(kite, symbol, option_type, spot_price, avoid_expiry_day=True):
    """
    Get the ATM (At-The-Money) option contract for a symbol.
    
    UPDATED: Now includes expiry day check and skips to next expiry if needed.
    
    Args:
        kite: KiteConnect instance
        symbol: Stock/Index symbol (e.g., 'RELIANCE', 'NIFTY')
        option_type: 'CE' for Call, 'PE' for Put
        spot_price: Current spot price of underlying
        avoid_expiry_day: If True, skip to next expiry on expiry day (default True)
        
    Returns:
        dict: Option contract details including tradingsymbol, token, strike, expiry, lot_size, days_to_expiry
        None: If no contract found
    """
    try:
        if st.session_state.nfo_instruments:
            instruments = st.session_state.nfo_instruments
        else:
            instruments = kite.instruments("NFO")
            st.session_state.nfo_instruments = instruments
        
        # Filter for this symbol and option type
        option_contracts = [
            inst for inst in instruments
            if symbol in inst['tradingsymbol']
            and inst['instrument_type'] == option_type
            and inst['expiry'] > datetime.now().date()
        ]
        
        if not option_contracts:
            return None
        
        # Sort by expiry
        option_contracts.sort(key=lambda x: x['expiry'])
        
        # Find appropriate expiry
        if avoid_expiry_day:
            # Skip expiry day
            selected_expiry = None
            for contract in option_contracts:
                if not is_expiry_day(contract['expiry']):
                    selected_expiry = contract['expiry']
                    break
            
            if selected_expiry is None:
                # All available expiries are today (shouldn't happen)
                return None
        else:
            selected_expiry = option_contracts[0]['expiry']
        
        # Filter for selected expiry
        expiry_contracts = [c for c in option_contracts if c['expiry'] == selected_expiry]
        
        # Find ATM strike
        strikes = [c['strike'] for c in expiry_contracts]
        
        if not strikes:
            return None
        
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # Get the ATM contract
        for contract in expiry_contracts:
            if contract['strike'] == atm_strike:
                days_to_exp = get_days_to_expiry(contract['expiry'])
                return {
                    'tradingsymbol': contract['tradingsymbol'],
                    'token': contract['instrument_token'],
                    'strike': contract['strike'],
                    'expiry': contract['expiry'],
                    'lot_size': contract['lot_size'],
                    'instrument_type': contract['instrument_type'],
                    'exchange': 'NFO',
                    'days_to_expiry': days_to_exp,
                    'is_expiry_day': days_to_exp == 0,
                    'expiry_warning': 'CAUTION: 1-2 DTE' if days_to_exp <= 2 else None
                }
        
        return None
        
    except Exception as e:
        print(f"Error getting ATM option contract: {e}")
        return None


def get_real_option_price(kite, symbol, option_type, spot_price, ws_manager=None, avoid_expiry_day=True):
    """
    Get REAL option price from market (not simulated).
    
    UPDATED: Now respects expiry day avoidance.
    
    Args:
        kite: KiteConnect instance
        symbol: Stock/Index symbol
        option_type: 'CE' or 'PE'
        spot_price: Current spot price
        ws_manager: WebSocketManager instance (optional)
        avoid_expiry_day: Skip expiry day contracts (default True)
        
    Returns:
        dict with premium, contract details, expiry info
        None: If unable to get price or all contracts are expiry day
    """
    try:
        # Get ATM option contract (with expiry day filtering)
        contract = get_atm_option_contract(kite, symbol, option_type, spot_price, avoid_expiry_day)
        
        if not contract:
            return None
        
        # Check if this is expiry day (shouldn't happen if avoid_expiry_day=True)
        if contract.get('is_expiry_day', False) and avoid_expiry_day:
            print(f"Skipping {symbol} - expiry day contract")
            return None
        
        token = contract['token']
        tradingsymbol = contract['tradingsymbol']
        
        # Try WebSocket first
        if ws_manager and ws_manager.is_connected:
            tick = ws_manager.get_tick(token)
            
            if tick and tick.get('ltp', 0) > 0:
                return {
                    'premium': tick['ltp'],
                    'contract': tradingsymbol,
                    'token': token,
                    'strike': contract['strike'],
                    'expiry': contract['expiry'],
                    'lot_size': contract['lot_size'],
                    'bid': tick.get('bid', 0),
                    'ask': tick.get('ask', 0),
                    'source': 'websocket',
                    'days_to_expiry': contract['days_to_expiry'],
                    'is_expiry_day': contract['is_expiry_day'],
                    'expiry_warning': contract.get('expiry_warning')
                }
        
        # Fallback to REST API
        ltp_data = kite.ltp(f"NFO:{tradingsymbol}")
        
        if ltp_data and f"NFO:{tradingsymbol}" in ltp_data:
            ltp = ltp_data[f"NFO:{tradingsymbol}"]['last_price']
            
            return {
                'premium': ltp,
                'contract': tradingsymbol,
                'token': token,
                'strike': contract['strike'],
                'expiry': contract['expiry'],
                'lot_size': contract['lot_size'],
                'bid': 0,
                'ask': 0,
                'source': 'rest',
                'days_to_expiry': contract['days_to_expiry'],
                'is_expiry_day': contract['is_expiry_day'],
                'expiry_warning': contract.get('expiry_warning')
            }
        
        return None
        
    except Exception as e:
        print(f"Error getting real option price: {e}")
        return None


def get_real_spot_price(kite, symbol, token, ws_manager=None):
    """
    Get REAL spot price for underlying.
    
    Args:
        kite: KiteConnect instance
        symbol: Stock/Index symbol
        token: Instrument token
        ws_manager: WebSocketManager instance (optional)
        
    Returns:
        float: Spot price
        0: If unable to get price
    """
    try:
        # Try WebSocket first
        if ws_manager and ws_manager.is_connected:
            ltp = ws_manager.get_ltp(token)
            if ltp > 0:
                return ltp
        
        # Fallback to REST
        exchange = 'NSE'
        ltp_data = kite.ltp(f"{exchange}:{symbol}")
        
        if ltp_data and f"{exchange}:{symbol}" in ltp_data:
            return ltp_data[f"{exchange}:{symbol}"]['last_price']
        
        return 0
        
    except Exception as e:
        print(f"Error getting spot price: {e}")
        return 0


def get_option_chain_snapshot(kite, symbol, spot_price, num_strikes=5):
    """
    Get a snapshot of the option chain around ATM.
    
    Args:
        kite: KiteConnect instance
        symbol: Stock/Index symbol
        spot_price: Current spot price
        num_strikes: Number of strikes above/below ATM
        
    Returns:
        DataFrame with option chain data
    """
    try:
        if st.session_state.nfo_instruments is None:
            st.session_state.nfo_instruments = kite.instruments("NFO")
        
        instruments = st.session_state.nfo_instruments
        
        # Filter for this symbol
        contracts = [
            inst for inst in instruments
            if symbol in inst['tradingsymbol']
            and inst['instrument_type'] in ['CE', 'PE']
            and inst['expiry'] > datetime.now().date()
        ]
        
        if not contracts:
            return None
        
        # Get nearest expiry (avoiding expiry day)
        contracts.sort(key=lambda x: x['expiry'])
        
        nearest_expiry = None
        for c in contracts:
            if not is_expiry_day(c['expiry']):
                nearest_expiry = c['expiry']
                break
        
        if nearest_expiry is None:
            nearest_expiry = contracts[0]['expiry']
        
        # Filter for nearest expiry
        contracts = [c for c in contracts if c['expiry'] == nearest_expiry]
        
        # Get unique strikes
        strikes = sorted(set(c['strike'] for c in contracts))
        
        # Find ATM index
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        atm_index = strikes.index(atm_strike)
        
        # Get strikes range
        start_idx = max(0, atm_index - num_strikes)
        end_idx = min(len(strikes), atm_index + num_strikes + 1)
        selected_strikes = strikes[start_idx:end_idx]
        
        # Build chain data
        chain_data = []
        
        for strike in selected_strikes:
            row = {'strike': strike}
            
            for contract in contracts:
                if contract['strike'] == strike:
                    option_type = contract['instrument_type']
                    
                    try:
                        ltp_data = kite.ltp(f"NFO:{contract['tradingsymbol']}")
                        ltp = ltp_data[f"NFO:{contract['tradingsymbol']}"]['last_price']
                    except:
                        ltp = 0
                    
                    row[f'{option_type}_premium'] = ltp
                    row[f'{option_type}_symbol'] = contract['tradingsymbol']
                    row[f'{option_type}_token'] = contract['instrument_token']
            
            chain_data.append(row)
        
        df = pd.DataFrame(chain_data)
        df['days_to_expiry'] = get_days_to_expiry(nearest_expiry)
        df['expiry'] = nearest_expiry
        
        return df
        
    except Exception as e:
        print(f"Error getting option chain: {e}")
        return None


def calculate_investment(premium, lot_size, num_lots=1):
    """
    Calculate total investment for an option trade.
    
    Formula: Premium × Lot Size × Number of Lots
    """
    return premium * lot_size * num_lots


def round_to_tick(price, tick_size=0.05):
    """
    Round price to nearest tick size.
    """
    return round(price / tick_size) * tick_size


def get_expiry_schedule():
    """
    Get information about upcoming expiries.
    
    Returns:
        dict with expiry information for major indices
    """
    today = datetime.now().date()
    
    # Standard expiry days (may vary, this is approximate)
    expiry_info = {
        'NIFTY': 'Thursday (Weekly)',
        'BANKNIFTY': 'Wednesday (Weekly)',
        'FINNIFTY': 'Tuesday (Weekly)',
        'MIDCPNIFTY': 'Monday (Weekly)',
        'stocks': 'Last Thursday of month'
    }
    
    return {
        'today': today,
        'is_any_expiry_today': today.weekday() in [0, 1, 2, 3],  # Mon-Thu
        'schedule': expiry_info
    }

# ============================================================================
# END OF CHUNK 4 (UPDATED)
# Changes Summary:
# 1. Added is_expiry_day() function
# 2. Added get_days_to_expiry() function
# 3. Added should_avoid_expiry() filter function
# 4. Added get_next_expiry_avoiding_current() function
# 5. Modified get_atm_option_contract() to include expiry filtering
# 6. Modified get_real_option_price() to respect expiry day avoidance
# 7. Added days_to_expiry to all contract return dicts
# 8. Added get_expiry_schedule() for user info
# ============================================================================


# ============================================================================
# CHUNK 5: TECHNICAL ANALYSIS CLASS (UPDATED)
# Description: Professional trading analysis with CLEANED indicators
# Changes Made:
#   - REMOVED: Williams %R, CCI, MFI (noise indicators)
#   - KEPT: RSI, MACD, Volume, Price Action, SMC
#   - INCREASED: Minimum confirmations from 4 to 6
#   - ADDED: Confirmation categories for diversity check
# Dependencies: Chunk 1-4
# ============================================================================

class ProfessionalTradingAnalysis:
    """
    Comprehensive technical analysis class for generating high-quality trading signals.
    
    CLEANED VERSION:
    - Removed weak/noisy indicators (Williams %R, CCI, MFI)
    - Focus on: Trend (EMA), Momentum (RSI, MACD, Stochastic), Volume, Structure (SMC)
    - Requires 6+ confirmations from DIFFERENT categories
    
    Usage:
        analyzer = ProfessionalTradingAnalysis(kite, 'RELIANCE', token, ws_manager)
        signal, confidence, analysis = analyzer.generate_professional_signal()
    """
    
    # Confirmation categories - signals must come from DIFFERENT categories
    CONFIRMATION_CATEGORIES = {
        'TREND': ['EMA_ALIGNMENT', 'PRICE_POSITION', 'TREND_DIRECTION'],
        'MOMENTUM': ['MACD', 'RSI', 'STOCH'],
        'VOLUME': ['VOLUME_SURGE', 'OBV', 'VOLUME_CONFIRM'],
        'VOLATILITY': ['BB_SIGNAL', 'ATR_CONFIRM'],
        'STRUCTURE': ['SUPPORT', 'RESISTANCE', 'VWAP'],
        'SMC': ['ORDER_BLOCK', 'BOS', 'FVG', 'LIQUIDITY_SWEEP'],
        'CANDLE': ['ENGULFING', 'HAMMER', 'SHOOTING_STAR'],
        'RISK_REWARD': ['RR_RATIO']
    }
    
    def __init__(self, kite, symbol, token, ws_manager=None):
        self.kite = kite
        self.symbol = symbol
        self.token = token
        self.ws_manager = ws_manager
    
    def get_historical_data(self, days=10, interval='5minute', specific_date=None):
        """
        Fetch historical OHLCV data.
        
        Args:
            days: Number of days of data
            interval: Candle interval ('minute', '5minute', '15minute', 'day')
            specific_date: Specific date for backtesting
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if specific_date:
                from_date = datetime.combine(specific_date, datetime.min.time())
                to_date = datetime.combine(specific_date, datetime.max.time())
            else:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days)
            
            data = self.kite.historical_data(
                instrument_token=self.token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def get_realtime_data(self):
        """
        Get real-time tick data from WebSocket.
        
        Returns:
            dict: Current tick data
        """
        if self.ws_manager and self.ws_manager.is_connected:
            return self.ws_manager.get_tick(self.token)
        return None
    
    def calculate_all_indicators(self, df):
        """
        Calculate CLEANED technical indicators.
        
        REMOVED (noise indicators):
        - Williams %R
        - CCI
        - MFI
        
        KEPT (proven indicators):
        - Trend: EMA (9, 21, 50, 200), MACD
        - Momentum: RSI, Stochastic
        - Volatility: Bollinger Bands, ATR
        - Volume: OBV, Volume SMA
        - VWAP, ADX
        """
        try:
            # === TREND INDICATORS ===
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
            
            df['macd'] = ta.trend.macd(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_diff'] = ta.trend.macd_diff(df['close'])
            
            # === MOMENTUM INDICATORS (CLEANED - removed Williams %R, CCI) ===
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            # REMOVED: df['cci'] = ta.trend.cci(...)
            # REMOVED: df['williams_r'] = ta.momentum.williams_r(...)
            
            # === VOLATILITY INDICATORS ===
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # === VOLUME INDICATORS (CLEANED - removed MFI) ===
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            # REMOVED: df['mfi'] = ta.volume.money_flow_index(...)
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # === VWAP ===
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # === ADX (Trend Strength) ===
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def identify_support_resistance(self, df, window=20):
        """
        Identify support and resistance levels using swing highs/lows.
        """
        try:
            support_levels = []
            resistance_levels = []
            
            for i in range(window, len(df) - window):
                # Swing low (potential support)
                if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                    support_levels.append(df['low'].iloc[i])
                
                # Swing high (potential resistance)
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                    resistance_levels.append(df['high'].iloc[i])
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            supports_below = [s for s in support_levels if s < current_price]
            resistances_above = [r for r in resistance_levels if r > current_price]
            
            nearest_support = max(supports_below) if supports_below else current_price * 0.97
            nearest_resistance = min(resistances_above) if resistances_above else current_price * 1.03
            
            return {
                'support_levels': sorted(set(support_levels), reverse=True)[:5],
                'resistance_levels': sorted(set(resistance_levels))[:5],
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'distance_to_support': ((current_price - nearest_support) / current_price * 100),
                'distance_to_resistance': ((nearest_resistance - current_price) / current_price * 100)
            }
            
        except Exception as e:
            return {}
    
    def smart_money_concepts(self, df):
        """
        Implement Smart Money Concepts (SMC) analysis.
        
        Identifies:
        - Order Blocks (institutional buying/selling zones)
        - Fair Value Gaps (imbalance areas)
        - Break of Structure (BOS)
        - Liquidity Sweeps
        """
        try:
            smc_signals = {
                'order_blocks': [],
                'fair_value_gaps': [],
                'liquidity_sweeps': [],
                'break_of_structure': False
            }
            
            if 'volume_sma' not in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # === ORDER BLOCKS ===
            for i in range(10, len(df)):
                # Bullish Order Block
                if (df['close'].iloc[i] > df['open'].iloc[i] and
                    df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                    df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.5):
                    
                    strength = 'HIGH' if df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 2 else 'MEDIUM'
                    smc_signals['order_blocks'].append({
                        'type': 'BULLISH',
                        'price': df['low'].iloc[i-1],
                        'strength': strength
                    })
                
                # Bearish Order Block
                if (df['close'].iloc[i] < df['open'].iloc[i] and
                    df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                    df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.5):
                    
                    strength = 'HIGH' if df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 2 else 'MEDIUM'
                    smc_signals['order_blocks'].append({
                        'type': 'BEARISH',
                        'price': df['high'].iloc[i-1],
                        'strength': strength
                    })
            
            # === FAIR VALUE GAPS ===
            for i in range(2, len(df)):
                # Bullish FVG
                if df['low'].iloc[i] > df['high'].iloc[i-2]:
                    smc_signals['fair_value_gaps'].append({
                        'type': 'BULLISH_FVG',
                        'top': df['low'].iloc[i],
                        'bottom': df['high'].iloc[i-2]
                    })
                
                # Bearish FVG
                if df['high'].iloc[i] < df['low'].iloc[i-2]:
                    smc_signals['fair_value_gaps'].append({
                        'type': 'BEARISH_FVG',
                        'top': df['low'].iloc[i-2],
                        'bottom': df['high'].iloc[i]
                    })
            
            # === BREAK OF STRUCTURE ===
            recent_highs = df['high'].tail(20)
            recent_lows = df['low'].tail(20)
            
            if df['close'].iloc[-1] > recent_highs.iloc[-10:-1].max():
                smc_signals['break_of_structure'] = 'BULLISH'
            elif df['close'].iloc[-1] < recent_lows.iloc[-10:-1].min():
                smc_signals['break_of_structure'] = 'BEARISH'
            
            # === LIQUIDITY SWEEPS ===
            for i in range(50, len(df)):
                # Bullish sweep
                if (df['low'].iloc[i] < df['low'].iloc[i-50:i].min() and
                    df['close'].iloc[i] > df['open'].iloc[i]):
                    smc_signals['liquidity_sweeps'].append({
                        'type': 'BULLISH_SWEEP',
                        'price': df['low'].iloc[i]
                    })
                
                # Bearish sweep
                if (df['high'].iloc[i] > df['high'].iloc[i-50:i].max() and
                    df['close'].iloc[i] < df['open'].iloc[i]):
                    smc_signals['liquidity_sweeps'].append({
                        'type': 'BEARISH_SWEEP',
                        'price': df['high'].iloc[i]
                    })
            
            return smc_signals
            
        except Exception as e:
            return {}
    
    def count_category_confirmations(self, confirmations):
        """
        Count how many DIFFERENT categories have confirmations.
        This ensures signal diversity - not just multiple signals from same category.
        
        Returns:
            int: Number of unique categories with confirmations
        """
        categories_hit = set()
        
        for confirmation in confirmations:
            for category, signals in self.CONFIRMATION_CATEGORIES.items():
                if confirmation in signals:
                    categories_hit.add(category)
                    break
        
        return len(categories_hit)
    
    def generate_professional_signal(self, df=None):
        """
        Generate trading signal with multiple confirmations and R-factor.
        
        UPDATED REQUIREMENTS:
        - Minimum 6 confirmations (up from 4)
        - Confirmations must come from DIFFERENT categories
        - Minimum score increased to 15 (up from 12)
        - Removed weak indicator signals (Williams %R, CCI, MFI)
        
        Returns:
            tuple: (signal, confidence, analysis_detail)
            - signal: 'CALL', 'PUT', or None
            - confidence: 0-100 score
            - analysis_detail: dict with all analysis data
        """
        # Get data if not provided
        if df is None:
            df = self.get_historical_data()
        
        if df is None or len(df) < 200:
            return None, 0, {'action': 'INSUFFICIENT_DATA'}
        
        # Calculate indicators
        df = self.calculate_all_indicators(df)
        
        current_price = df['close'].iloc[-1]
        
        # Get analysis components
        sr_levels = self.identify_support_resistance(df)
        smc = self.smart_money_concepts(df)
        
        # Initialize scores
        buy_score = 0
        sell_score = 0
        
        signals_detail = {
            'price': current_price,
            'indicators': [],
            'confirmations': [],  # Now stores category-tagged confirmations
            'strength': 0,
            'zones': sr_levels,
            'smc': smc,
            'r_factor': 0,
            'category_count': 0  # NEW: Track unique categories
        }
        
        # === TREND ANALYSIS (Weight: 4) ===
        ema_9 = df['ema_9'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1]
        
        if ema_9 > ema_21 > ema_50 > ema_200:
            buy_score += 4
            signals_detail['indicators'].append('💪 Perfect Bullish Alignment')
            signals_detail['confirmations'].append('EMA_ALIGNMENT')
            signals_detail['confirmations'].append('TREND_DIRECTION')
        elif ema_9 < ema_21 < ema_50 < ema_200:
            sell_score += 4
            signals_detail['indicators'].append('💪 Perfect Bearish Alignment')
            signals_detail['confirmations'].append('EMA_ALIGNMENT')
            signals_detail['confirmations'].append('TREND_DIRECTION')
        elif ema_9 > ema_21 > ema_50:
            buy_score += 2.5
            signals_detail['indicators'].append('Bullish Trend')
            signals_detail['confirmations'].append('TREND_DIRECTION')
        elif ema_9 < ema_21 < ema_50:
            sell_score += 2.5
            signals_detail['indicators'].append('Bearish Trend')
            signals_detail['confirmations'].append('TREND_DIRECTION')
        
        # Price position relative to EMAs
        if current_price > ema_9 > ema_21 > ema_50:
            buy_score += 1.5
            signals_detail['confirmations'].append('PRICE_POSITION')
        elif current_price < ema_9 < ema_21 < ema_50:
            sell_score += 1.5
            signals_detail['confirmations'].append('PRICE_POSITION')
        
        # === MOMENTUM (Weight: 3.5) - CLEANED ===
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_diff = df['macd_diff'].iloc[-1]
        macd_diff_prev = df['macd_diff'].iloc[-2]
        
        # MACD
        if macd > macd_signal and macd_diff > 0 and macd_diff > macd_diff_prev:
            buy_score += 3
            signals_detail['indicators'].append('MACD Strong Bullish')
            signals_detail['confirmations'].append('MACD')
        elif macd < macd_signal and macd_diff < 0 and macd_diff < macd_diff_prev:
            sell_score += 3
            signals_detail['indicators'].append('MACD Strong Bearish')
            signals_detail['confirmations'].append('MACD')
        
        # RSI
        if 30 < rsi < 40 and rsi > rsi_prev:
            buy_score += 2.5
            signals_detail['indicators'].append(f'RSI Reversal Zone ({rsi:.1f})')
            signals_detail['confirmations'].append('RSI')
        elif 60 < rsi < 70 and rsi < rsi_prev:
            sell_score += 2.5
            signals_detail['indicators'].append(f'RSI Reversal Zone ({rsi:.1f})')
            signals_detail['confirmations'].append('RSI')
        elif rsi < 30:
            buy_score += 1.5
            signals_detail['indicators'].append(f'RSI Oversold ({rsi:.1f})')
            signals_detail['confirmations'].append('RSI')
        elif rsi > 70:
            sell_score += 1.5
            signals_detail['indicators'].append(f'RSI Overbought ({rsi:.1f})')
            signals_detail['confirmations'].append('RSI')
        
        # Stochastic
        stoch = df['stoch'].iloc[-1]
        stoch_signal = df['stoch_signal'].iloc[-1]
        
        if stoch > stoch_signal and stoch < 80 and stoch > 20:
            buy_score += 1.5
            signals_detail['confirmations'].append('STOCH')
            signals_detail['indicators'].append(f'Stoch Bullish ({stoch:.1f})')
        elif stoch < stoch_signal and stoch > 20 and stoch < 80:
            sell_score += 1.5
            signals_detail['confirmations'].append('STOCH')
            signals_detail['indicators'].append(f'Stoch Bearish ({stoch:.1f})')
        
        # REMOVED: Williams %R analysis
        # REMOVED: CCI analysis
        # REMOVED: MFI analysis
        
        # === VOLUME (Weight: 3) - CLEANED ===
        volume = df['volume'].iloc[-1]
        volume_sma = df['volume_sma'].iloc[-1]
        volume_prev = df['volume'].iloc[-2]
        obv = df['obv'].iloc[-1]
        obv_prev = df['obv'].iloc[-5]
        
        if volume > volume_sma * 2 and volume > volume_prev:
            if current_price > df['close'].iloc[-2]:
                buy_score += 3
                signals_detail['indicators'].append('🔥 Volume Surge Bullish')
                signals_detail['confirmations'].append('VOLUME_SURGE')
            else:
                sell_score += 3
                signals_detail['indicators'].append('🔥 Volume Surge Bearish')
                signals_detail['confirmations'].append('VOLUME_SURGE')
        elif volume > volume_sma * 1.5:
            if current_price > df['close'].iloc[-2]:
                buy_score += 1.5
                signals_detail['indicators'].append('High Volume Bullish')
                signals_detail['confirmations'].append('VOLUME_CONFIRM')
            else:
                sell_score += 1.5
                signals_detail['indicators'].append('High Volume Bearish')
                signals_detail['confirmations'].append('VOLUME_CONFIRM')
        
        # OBV trend
        if obv > obv_prev:
            buy_score += 1.5
            signals_detail['confirmations'].append('OBV')
            signals_detail['indicators'].append('OBV Rising')
        elif obv < obv_prev:
            sell_score += 1.5
            signals_detail['confirmations'].append('OBV')
            signals_detail['indicators'].append('OBV Falling')
        
        # REMOVED: MFI analysis
        
        # === VOLATILITY (Weight: 2.5) ===
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        atr = df['atr'].iloc[-1]
        atr_sma = df['atr'].rolling(window=14).mean().iloc[-1]
        
        if current_price <= bb_lower and current_price > df['close'].iloc[-2]:
            buy_score += 2.5
            signals_detail['indicators'].append('BB Lower Bounce')
            signals_detail['confirmations'].append('BB_SIGNAL')
        elif current_price >= bb_upper and current_price < df['close'].iloc[-2]:
            sell_score += 2.5
            signals_detail['indicators'].append('BB Upper Rejection')
            signals_detail['confirmations'].append('BB_SIGNAL')
        
        # ATR confirmation
        if atr < atr_sma * 0.8:
            signals_detail['indicators'].append('Low Volatility (Breakout Setup)')
            signals_detail['confirmations'].append('ATR_CONFIRM')
        
        # === VWAP (Weight: 2) ===
        vwap = df['vwap'].iloc[-1]
        if current_price > vwap * 1.01:
            buy_score += 2
            signals_detail['indicators'].append('Above VWAP')
            signals_detail['confirmations'].append('VWAP')
        elif current_price < vwap * 0.99:
            sell_score += 2
            signals_detail['indicators'].append('Below VWAP')
            signals_detail['confirmations'].append('VWAP')
        
        # === ADX (Weight: 2.5) ===
        adx = df['adx'].iloc[-1]
        if adx > 30:
            signals_detail['indicators'].append(f'Very Strong Trend (ADX: {adx:.1f})')
            if buy_score > sell_score:
                buy_score *= 1.3
            else:
                sell_score *= 1.3
        elif adx > 25:
            signals_detail['indicators'].append(f'Strong Trend (ADX: {adx:.1f})')
            if buy_score > sell_score:
                buy_score *= 1.15
            else:
                sell_score *= 1.15
        elif adx < 20:
            buy_score *= 0.7
            sell_score *= 0.7
            signals_detail['indicators'].append(f'⚠️ Weak Trend (ADX: {adx:.1f}) - AVOID')
        
        # === SUPPORT/RESISTANCE (Weight: 3) ===
        if sr_levels:
            dist_support = sr_levels.get('distance_to_support', 100)
            dist_resistance = sr_levels.get('distance_to_resistance', 100)
            
            if dist_support < 1.5:
                buy_score += 3
                signals_detail['indicators'].append(f'At Support ({dist_support:.1f}%)')
                signals_detail['confirmations'].append('SUPPORT')
            elif dist_support < 3:
                buy_score += 1.5
                signals_detail['indicators'].append(f'Near Support ({dist_support:.1f}%)')
            
            if dist_resistance < 1.5:
                sell_score += 3
                signals_detail['indicators'].append(f'At Resistance ({dist_resistance:.1f}%)')
                signals_detail['confirmations'].append('RESISTANCE')
            elif dist_resistance < 3:
                sell_score += 1.5
                signals_detail['indicators'].append(f'Near Resistance ({dist_resistance:.1f}%)')
        
        # === SMART MONEY CONCEPTS (Weight: 3.5) ===
        if smc:
            bullish_obs = [ob for ob in smc.get('order_blocks', []) 
                         if ob['type'] == 'BULLISH' and ob['strength'] == 'HIGH']
            bearish_obs = [ob for ob in smc.get('order_blocks', []) 
                         if ob['type'] == 'BEARISH' and ob['strength'] == 'HIGH']
            
            if bullish_obs:
                buy_score += 3
                signals_detail['indicators'].append('High Quality Bullish OB')
                signals_detail['confirmations'].append('ORDER_BLOCK')
            
            if bearish_obs:
                sell_score += 3
                signals_detail['indicators'].append('High Quality Bearish OB')
                signals_detail['confirmations'].append('ORDER_BLOCK')
            
            bos = smc.get('break_of_structure', False)
            if bos == 'BULLISH':
                buy_score += 2
                signals_detail['indicators'].append('Bullish BOS')
                signals_detail['confirmations'].append('BOS')
            elif bos == 'BEARISH':
                sell_score += 2
                signals_detail['indicators'].append('Bearish BOS')
                signals_detail['confirmations'].append('BOS')
            
            # FVG check
            bullish_fvg = [f for f in smc.get('fair_value_gaps', []) if f['type'] == 'BULLISH_FVG']
            bearish_fvg = [f for f in smc.get('fair_value_gaps', []) if f['type'] == 'BEARISH_FVG']
            
            if bullish_fvg and buy_score > sell_score:
                buy_score += 1.5
                signals_detail['confirmations'].append('FVG')
                signals_detail['indicators'].append('Bullish FVG Present')
            
            if bearish_fvg and sell_score > buy_score:
                sell_score += 1.5
                signals_detail['confirmations'].append('FVG')
                signals_detail['indicators'].append('Bearish FVG Present')
            
            # Liquidity sweep
            recent_sweeps = smc.get('liquidity_sweeps', [])[-3:]
            bullish_sweeps = [s for s in recent_sweeps if s['type'] == 'BULLISH_SWEEP']
            bearish_sweeps = [s for s in recent_sweeps if s['type'] == 'BEARISH_SWEEP']
            
            if bullish_sweeps:
                buy_score += 1.5
                signals_detail['confirmations'].append('LIQUIDITY_SWEEP')
                signals_detail['indicators'].append('Bullish Liquidity Sweep')
            
            if bearish_sweeps:
                sell_score += 1.5
                signals_detail['confirmations'].append('LIQUIDITY_SWEEP')
                signals_detail['indicators'].append('Bearish Liquidity Sweep')
        
        # === R-FACTOR CALCULATION ===
        if sr_levels:
            nearest_support = sr_levels.get('nearest_support', current_price * 0.97)
            nearest_resistance = sr_levels.get('nearest_resistance', current_price * 1.03)
            
            if buy_score > sell_score:
                potential_risk = abs(current_price - nearest_support)
                potential_reward = abs(nearest_resistance - current_price)
                
                if potential_risk > 0:
                    r_factor = potential_reward / potential_risk
                    signals_detail['r_factor'] = round(r_factor, 2)
                    
                    if r_factor >= 2.5:
                        buy_score += 4
                        signals_detail['indicators'].append(f'🎯 Excellent R:R ({r_factor:.1f}:1)')
                        signals_detail['confirmations'].append('RR_RATIO')
                    elif r_factor >= 2:
                        buy_score += 3
                        signals_detail['indicators'].append(f'Good R:R ({r_factor:.1f}:1)')
                        signals_detail['confirmations'].append('RR_RATIO')
                    elif r_factor >= 1.5:
                        buy_score += 1.5
                        signals_detail['indicators'].append(f'Acceptable R:R ({r_factor:.1f}:1)')
                    elif r_factor < 1.5:
                        buy_score *= 0.4  # Heavy penalty for poor R:R
                        signals_detail['indicators'].append(f'⚠️ Poor R:R ({r_factor:.1f}:1) - SKIP')
            
            elif sell_score > buy_score:
                potential_risk = abs(nearest_resistance - current_price)
                potential_reward = abs(current_price - nearest_support)
                
                if potential_risk > 0:
                    r_factor = potential_reward / potential_risk
                    signals_detail['r_factor'] = round(r_factor, 2)
                    
                    if r_factor >= 2.5:
                        sell_score += 4
                        signals_detail['indicators'].append(f'🎯 Excellent R:R ({r_factor:.1f}:1)')
                        signals_detail['confirmations'].append('RR_RATIO')
                    elif r_factor >= 2:
                        sell_score += 3
                        signals_detail['indicators'].append(f'Good R:R ({r_factor:.1f}:1)')
                        signals_detail['confirmations'].append('RR_RATIO')
                    elif r_factor >= 1.5:
                        sell_score += 1.5
                        signals_detail['indicators'].append(f'Acceptable R:R ({r_factor:.1f}:1)')
                    elif r_factor < 1.5:
                        sell_score *= 0.4  # Heavy penalty for poor R:R
                        signals_detail['indicators'].append(f'⚠️ Poor R:R ({r_factor:.1f}:1) - SKIP')
        
        # === CANDLE PATTERNS ===
        close_curr = df['close'].iloc[-1]
        open_curr = df['open'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        open_prev = df['open'].iloc[-2]
        high_curr = df['high'].iloc[-1]
        low_curr = df['low'].iloc[-1]
        
        # Bullish engulfing
        if (close_curr > open_curr and close_prev < open_prev and
            close_curr > open_prev and open_curr < close_prev):
            buy_score += 2
            signals_detail['indicators'].append('Bullish Engulfing')
            signals_detail['confirmations'].append('ENGULFING')
        
        # Bearish engulfing
        if (close_curr < open_curr and close_prev > open_prev and
            close_curr < open_prev and open_curr > close_prev):
            sell_score += 2
            signals_detail['indicators'].append('Bearish Engulfing')
            signals_detail['confirmations'].append('ENGULFING')
        
        # Hammer
        body = abs(close_curr - open_curr)
        lower_wick = min(close_curr, open_curr) - low_curr
        upper_wick = high_curr - max(close_curr, open_curr)
        
        if body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5:
            buy_score += 1.5
            signals_detail['indicators'].append('Hammer Pattern')
            signals_detail['confirmations'].append('HAMMER')
        
        # Shooting star
        if body > 0 and upper_wick > body * 2 and lower_wick < body * 0.5:
            sell_score += 1.5
            signals_detail['indicators'].append('Shooting Star')
            signals_detail['confirmations'].append('SHOOTING_STAR')
        
        # === FINAL DECISION (UPDATED REQUIREMENTS) ===
        signal = None
        confidence = 0
        
        # UPDATED: Increased minimums
        min_score = 15  # UP from 12
        min_confirmations = 6  # UP from 4
        min_categories = 4  # NEW: Must have signals from at least 4 different categories
        min_r_factor = 1.5  # UP from 1.2
        
        # Count unique categories
        category_count = self.count_category_confirmations(signals_detail['confirmations'])
        
        signals_detail['buy_score'] = round(buy_score, 2)
        signals_detail['sell_score'] = round(sell_score, 2)
        signals_detail['confirmation_count'] = len(signals_detail['confirmations'])
        signals_detail['category_count'] = category_count
        signals_detail['unique_confirmations'] = list(set(signals_detail['confirmations']))
        
        # Debug info
        signals_detail['requirements'] = {
            'min_score': min_score,
            'min_confirmations': min_confirmations,
            'min_categories': min_categories,
            'min_r_factor': min_r_factor
        }
        
        if buy_score >= min_score and buy_score > sell_score * 1.8:
            if len(signals_detail['confirmations']) >= min_confirmations:
                if category_count >= min_categories:
                    if signals_detail['r_factor'] >= min_r_factor or signals_detail['r_factor'] == 0:
                        signal = 'CALL'
                        confidence = min(buy_score * 5, 100)  # Adjusted multiplier
                        signals_detail['action'] = 'BUY CALL'
                        signals_detail['strength'] = 'HIGH' if buy_score >= 20 else 'MODERATE'
                    else:
                        signals_detail['action'] = f'POOR_RISK_REWARD (R:R {signals_detail["r_factor"]:.1f} < {min_r_factor})'
                else:
                    signals_detail['action'] = f'INSUFFICIENT_CATEGORIES ({category_count} < {min_categories})'
            else:
                signals_detail['action'] = f'WEAK_SIGNAL ({len(signals_detail["confirmations"])} < {min_confirmations} confirmations)'
        
        elif sell_score >= min_score and sell_score > buy_score * 1.8:
            if len(signals_detail['confirmations']) >= min_confirmations:
                if category_count >= min_categories:
                    if signals_detail['r_factor'] >= min_r_factor or signals_detail['r_factor'] == 0:
                        signal = 'PUT'
                        confidence = min(sell_score * 5, 100)  # Adjusted multiplier
                        signals_detail['action'] = 'BUY PUT'
                        signals_detail['strength'] = 'HIGH' if sell_score >= 20 else 'MODERATE'
                    else:
                        signals_detail['action'] = f'POOR_RISK_REWARD (R:R {signals_detail["r_factor"]:.1f} < {min_r_factor})'
                else:
                    signals_detail['action'] = f'INSUFFICIENT_CATEGORIES ({category_count} < {min_categories})'
            else:
                signals_detail['action'] = f'WEAK_SIGNAL ({len(signals_detail["confirmations"])} < {min_confirmations} confirmations)'
        else:
            if buy_score < min_score and sell_score < min_score:
                signals_detail['action'] = f'NO_SIGNAL (scores: buy={buy_score:.1f}, sell={sell_score:.1f} < {min_score})'
            else:
                signals_detail['action'] = f'CONFLICTING_SIGNALS (buy={buy_score:.1f}, sell={sell_score:.1f})'
        
        return signal, confidence, signals_detail

# ============================================================================
# END OF CHUNK 5 (UPDATED)
# Changes Summary:
# 1. Removed Williams %R, CCI, MFI indicators
# 2. Added category-based confirmation tracking
# 3. Increased minimum score from 12 to 15
# 4. Increased minimum confirmations from 4 to 6
# 5. Added minimum 4 different categories requirement
# 6. Increased minimum R:R from 1.2 to 1.5
# 7. Added detailed rejection reasons for debugging
# ============================================================================



# ============================================================================
# CHUNK 6: TRADE EXECUTION FUNCTIONS (UPDATED - GTT OCO)
# Description: Functions for executing paper and live trades with REAL prices
# Changes Made:
#   - REPLACED: Separate GTT Target + GTT SL with single GTT OCO order
#   - ADDED: get_gtt_orders() function to fetch all GTT orders
#   - ADDED: get_gtt_status() function to check GTT status
#   - ADDED: cancel_gtt_order() function
#   - UPDATED: place_live_trade() to use GTT OCO
#   - UPDATED: update_live_positions() for trailing with OCO
#   - UPDATED: close_live_position() to handle OCO cancellation
# Dependencies: Chunk 1-5
# ============================================================================

def place_paper_trade(kite, symbol, signal, spot_price, num_lots, stock_info, analysis_detail, ws_manager=None):
    """
    Execute paper trade with REAL option prices.
    """
    try:
        lot_size = stock_info.get('lot_size', 1)
        option_type = 'CE' if signal == 'CALL' else 'PE'
        
        # Get REAL option price
        option_data = get_real_option_price(kite, symbol, option_type, spot_price, ws_manager)
        
        if not option_data:
            log = f"{datetime.now().strftime('%H:%M:%S')} | PAPER FAILED | {symbol} | Could not get option price"
            st.session_state.trade_logs.append(log)
            return None
        
        premium = option_data['premium']
        contract = option_data['contract']
        option_token = option_data['token']
        strike = option_data['strike']
        expiry = option_data['expiry']
        
        # Calculate investment: Premium × Lot Size × Number of Lots
        total_quantity = num_lots * lot_size
        investment = premium * total_quantity
        
        # Check if enough capital
        if investment > st.session_state.paper_capital:
            log = f"{datetime.now().strftime('%H:%M:%S')} | PAPER FAILED | {symbol} | Insufficient capital"
            st.session_state.trade_logs.append(log)
            return None
        
        # Create trade record
        trade = {
            'id': len(st.session_state.paper_positions) + 1,
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'contract': contract,
            'option_token': option_token,
            'strike': strike,
            'expiry': expiry,
            'spot_price': spot_price,
            'entry_price': premium,
            'current_price': premium,
            'num_lots': num_lots,
            'lot_size': lot_size,
            'total_qty': total_quantity,
            'investment': investment,
            'current_value': investment,
            'pnl': 0,
            'pnl_pct': 0,
            'status': 'OPEN',
            'confidence': analysis_detail.get('strength', 'MODERATE'),
            'indicators': analysis_detail.get('indicators', [])[:3],
            'max_profit': 0,
            'max_loss': 0,
            'price_source': option_data['source']
        }
        
        st.session_state.paper_positions.append(trade)
        st.session_state.paper_capital -= investment
        
        log = f"{datetime.now().strftime('%H:%M:%S')} | ✅ PAPER TRADE | {symbol} {signal} | Contract: {contract} | Premium: ₹{premium:.2f} × {total_quantity} = ₹{investment:.2f}"
        st.session_state.trade_logs.append(log)
        
        return trade
        
    except Exception as e:
        log = f"{datetime.now().strftime('%H:%M:%S')} | PAPER ERROR | {symbol} | {str(e)}"
        st.session_state.trade_logs.append(log)
        return None


def update_paper_positions(kite, target_pct, stop_loss_pct, ws_manager=None):
    """Update paper positions with REAL current option prices."""
    for position in st.session_state.paper_positions[:]:
        if position['status'] != 'OPEN':
            continue
        
        try:
            option_token = position.get('option_token')
            contract = position['contract']
            
            current_price = 0
            
            # Try WebSocket first
            if ws_manager and ws_manager.is_connected and option_token:
                tick = ws_manager.get_tick(option_token)
                if tick and tick.get('ltp', 0) > 0:
                    current_price = tick['ltp']
                    position['price_source'] = 'websocket'
            
            # Fallback to REST
            if current_price == 0:
                try:
                    ltp_data = kite.ltp(f"NFO:{contract}")
                    if ltp_data and f"NFO:{contract}" in ltp_data:
                        current_price = ltp_data[f"NFO:{contract}"]['last_price']
                        position['price_source'] = 'rest'
                except:
                    pass
            
            if current_price == 0:
                continue
            
            position['current_price'] = current_price
            position['current_value'] = current_price * position['total_qty']
            position['pnl'] = position['current_value'] - position['investment']
            position['pnl_pct'] = (position['pnl'] / position['investment']) * 100
            
            if position['pnl'] > position['max_profit']:
                position['max_profit'] = position['pnl']
            if position['pnl'] < position['max_loss']:
                position['max_loss'] = position['pnl']
            
            # Check exit conditions
            should_exit = False
            exit_reason = ''
            
            if position['pnl_pct'] >= target_pct:
                should_exit = True
                exit_reason = f'Target Hit ({position["pnl_pct"]:.1f}%)'
            elif position['pnl_pct'] <= -stop_loss_pct:
                should_exit = True
                exit_reason = f'Stop Loss Hit ({position["pnl_pct"]:.1f}%)'
            
            if should_exit:
                close_paper_position(position, exit_reason)
        except:
            pass


def close_paper_position(position, reason):
    """Close a paper trading position."""
    position['status'] = 'CLOSED'
    position['exit_time'] = datetime.now()
    position['exit_reason'] = reason
    position['exit_price'] = position['current_price']
    
    st.session_state.paper_capital += position['current_value']
    
    st.session_state.performance_stats['total_trades'] += 1
    st.session_state.performance_stats['total_pnl'] += position['pnl']
    
    if position['pnl'] > 0:
        st.session_state.performance_stats['winning_trades'] += 1
        if position['pnl'] > st.session_state.performance_stats['best_trade']:
            st.session_state.performance_stats['best_trade'] = position['pnl']
    else:
        st.session_state.performance_stats['losing_trades'] += 1
        if position['pnl'] < st.session_state.performance_stats['worst_trade']:
            st.session_state.performance_stats['worst_trade'] = position['pnl']
    
    total = st.session_state.performance_stats['total_trades']
    wins = st.session_state.performance_stats['winning_trades']
    st.session_state.performance_stats['win_rate'] = (wins / total * 100) if total > 0 else 0
    
    log = f"{datetime.now().strftime('%H:%M:%S')} | 📤 PAPER CLOSED | {position['symbol']} | P&L: ₹{position['pnl']:.2f} ({position['pnl_pct']:.1f}%) | {reason}"
    st.session_state.trade_logs.append(log)


def verify_order_filled(kite, order_id, max_retries=10, sleep_time=2):
    """Verify if order is filled."""
    for i in range(max_retries):
        try:
            order_history = kite.order_history(order_id)
            if order_history:
                latest_status = order_history[-1]['status']
                if latest_status == 'COMPLETE':
                    return True, order_history[-1]
                elif latest_status in ['REJECTED', 'CANCELLED']:
                    return False, order_history[-1]
            time.sleep(sleep_time)
        except:
            time.sleep(sleep_time)
    return False, None


# ============================================================================
# GTT OCO FUNCTIONS (NEW)
# ============================================================================

def place_gtt_oco(kite, tradingsymbol, exchange, quantity, entry_price, target_price, sl_price):
    """
    Place a GTT OCO (One Cancels Other) order.
    
    GTT OCO places both target and stop loss in a single order.
    When one triggers, the other is automatically cancelled.
    
    Args:
        kite: KiteConnect instance
        tradingsymbol: Option contract symbol
        exchange: Exchange (NFO)
        quantity: Total quantity
        entry_price: Entry price (LTP at time of order)
        target_price: Target exit price
        sl_price: Stop loss price
        
    Returns:
        dict: {'gtt_id': id, 'status': 'success'/'failed', 'message': str}
    """
    try:
        # GTT OCO requires two trigger values and two orders
        # First trigger = Stop Loss (lower price)
        # Second trigger = Target (higher price)
        
        gtt_id = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_OCO,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            trigger_values=[sl_price, target_price],  # [SL trigger, Target trigger]
            last_price=entry_price,
            orders=[
                # Order 1: Stop Loss Order (triggers when price <= sl_price)
                {
                    'exchange': exchange,
                    'tradingsymbol': tradingsymbol,
                    'transaction_type': kite.TRANSACTION_TYPE_SELL,
                    'quantity': quantity,
                    'order_type': kite.ORDER_TYPE_LIMIT,
                    'product': kite.PRODUCT_MIS,
                    'price': sl_price  # Limit price for SL
                },
                # Order 2: Target Order (triggers when price >= target_price)
                {
                    'exchange': exchange,
                    'tradingsymbol': tradingsymbol,
                    'transaction_type': kite.TRANSACTION_TYPE_SELL,
                    'quantity': quantity,
                    'order_type': kite.ORDER_TYPE_LIMIT,
                    'product': kite.PRODUCT_MIS,
                    'price': target_price  # Limit price for Target
                }
            ]
        )
        
        return {
            'gtt_id': gtt_id,
            'status': 'success',
            'message': f'GTT OCO placed successfully. ID: {gtt_id}',
            'target_price': target_price,
            'sl_price': sl_price
        }
        
    except Exception as e:
        return {
            'gtt_id': None,
            'status': 'failed',
            'message': f'GTT OCO failed: {str(e)}',
            'target_price': target_price,
            'sl_price': sl_price
        }


def get_gtt_orders(kite):
    """
    Fetch all GTT orders from Zerodha.
    
    Returns:
        list: List of GTT orders with details
    """
    try:
        gtts = kite.get_gtts()
        
        # Process and format GTT data
        formatted_gtts = []
        for gtt in gtts:
            formatted_gtt = {
                'id': gtt.get('id'),
                'tradingsymbol': gtt.get('condition', {}).get('tradingsymbol', 'N/A'),
                'exchange': gtt.get('condition', {}).get('exchange', 'N/A'),
                'trigger_type': gtt.get('type', 'N/A'),  # 'single' or 'two-leg' (OCO)
                'status': gtt.get('status', 'N/A'),
                'created_at': gtt.get('created_at', 'N/A'),
                'updated_at': gtt.get('updated_at', 'N/A'),
                'expires_at': gtt.get('expires_at', 'N/A'),
                'trigger_values': gtt.get('condition', {}).get('trigger_values', []),
                'last_price': gtt.get('condition', {}).get('last_price', 0),
                'orders': gtt.get('orders', []),
                'meta': gtt.get('meta', {})
            }
            
            # Determine if it's OCO or Single
            if gtt.get('type') == 'two-leg':
                formatted_gtt['gtt_type'] = 'OCO'
                if len(formatted_gtt['trigger_values']) >= 2:
                    formatted_gtt['sl_trigger'] = formatted_gtt['trigger_values'][0]
                    formatted_gtt['target_trigger'] = formatted_gtt['trigger_values'][1]
            else:
                formatted_gtt['gtt_type'] = 'SINGLE'
                if formatted_gtt['trigger_values']:
                    formatted_gtt['trigger_price'] = formatted_gtt['trigger_values'][0]
            
            # Get order quantities and prices
            if formatted_gtt['orders']:
                formatted_gtt['quantity'] = formatted_gtt['orders'][0].get('quantity', 0)
                formatted_gtt['order_prices'] = [o.get('price', 0) for o in formatted_gtt['orders']]
            
            formatted_gtts.append(formatted_gtt)
        
        return formatted_gtts
        
    except Exception as e:
        print(f"Error fetching GTT orders: {e}")
        return []


def get_gtt_status(kite, gtt_id):
    """
    Get status of a specific GTT order.
    
    Args:
        kite: KiteConnect instance
        gtt_id: GTT order ID
        
    Returns:
        dict: GTT status details
    """
    try:
        gtts = kite.get_gtts()
        
        for gtt in gtts:
            if gtt.get('id') == gtt_id:
                return {
                    'found': True,
                    'id': gtt_id,
                    'status': gtt.get('status', 'unknown'),
                    'type': gtt.get('type', 'unknown'),
                    'tradingsymbol': gtt.get('condition', {}).get('tradingsymbol', 'N/A'),
                    'trigger_values': gtt.get('condition', {}).get('trigger_values', []),
                    'created_at': gtt.get('created_at'),
                    'updated_at': gtt.get('updated_at')
                }
        
        return {
            'found': False,
            'id': gtt_id,
            'status': 'not_found',
            'message': 'GTT order not found - may have been triggered or cancelled'
        }
        
    except Exception as e:
        return {
            'found': False,
            'id': gtt_id,
            'status': 'error',
            'message': f'Error fetching GTT status: {str(e)}'
        }


def cancel_gtt_order(kite, gtt_id):
    """
    Cancel a GTT order.
    
    Args:
        kite: KiteConnect instance
        gtt_id: GTT order ID to cancel
        
    Returns:
        dict: {'success': bool, 'message': str}
    """
    try:
        kite.delete_gtt(gtt_id)
        return {
            'success': True,
            'message': f'GTT {gtt_id} cancelled successfully'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to cancel GTT {gtt_id}: {str(e)}'
        }


def modify_gtt_oco(kite, old_gtt_id, tradingsymbol, exchange, quantity, current_price, new_target, new_sl):
    """
    Modify GTT OCO by cancelling old and placing new.
    
    Note: Zerodha doesn't support direct GTT modification,
    so we cancel and replace.
    
    Args:
        kite: KiteConnect instance
        old_gtt_id: Existing GTT ID to cancel
        tradingsymbol: Option contract symbol
        exchange: Exchange
        quantity: Quantity
        current_price: Current LTP
        new_target: New target price
        new_sl: New stop loss price
        
    Returns:
        dict: Result with new GTT ID or error
    """
    try:
        # Step 1: Cancel old GTT
        cancel_result = cancel_gtt_order(kite, old_gtt_id)
        
        if not cancel_result['success']:
            # GTT might already be triggered/cancelled, proceed anyway
            pass
        
        time.sleep(0.5)  # Brief pause
        
        # Step 2: Place new GTT OCO
        new_gtt_result = place_gtt_oco(
            kite=kite,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            quantity=quantity,
            entry_price=current_price,
            target_price=new_target,
            sl_price=new_sl
        )
        
        return {
            'success': new_gtt_result['status'] == 'success',
            'old_gtt_id': old_gtt_id,
            'new_gtt_id': new_gtt_result.get('gtt_id'),
            'new_target': new_target,
            'new_sl': new_sl,
            'message': new_gtt_result['message']
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'GTT modification failed: {str(e)}'
        }


# ============================================================================
# UPDATED LIVE TRADING FUNCTIONS (Using GTT OCO)
# ============================================================================

def place_live_trade(kite, symbol, signal, spot_price, quantity, stock_info, analysis_detail, 
                     target_pct, stop_loss_pct, ws_manager=None):
    """
    Execute LIVE trade on Zerodha with GTT OCO order.
    
    UPDATED: Now uses single GTT OCO instead of separate Target + SL GTTs.
    
    Benefits of GTT OCO:
    - Single order manages both target and stop loss
    - When one triggers, the other is automatically cancelled
    - Cleaner position management
    - Reduced GTT count (important as Zerodha has GTT limits)
    """
    try:
        lot_size = stock_info.get('lot_size', 1)
        option_type = 'CE' if signal == 'CALL' else 'PE'
        
        contract = get_atm_option_contract(kite, symbol, option_type, spot_price)
        
        if not contract:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ❌ LIVE FAILED | {symbol} | No option contract"
            st.session_state.trade_logs.append(log)
            return None
        
        tradingsymbol = contract['tradingsymbol']
        option_token = contract['token']
        
        # Get current LTP
        ltp_data = kite.ltp(f"NFO:{tradingsymbol}")
        ltp_price = ltp_data[f"NFO:{tradingsymbol}"]['last_price']
        
        order_price = round_to_tick(ltp_price - 0.25)  # Slightly below for better fill
        total_qty = quantity * lot_size
        
        # Place entry order
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=total_qty,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=order_price,
            validity=kite.VALIDITY_DAY
        )
        
        # Verify order filled
        is_filled, order_details = verify_order_filled(kite, order_id)
        
        if not is_filled:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ❌ ORDER NOT FILLED | {symbol}"
            st.session_state.trade_logs.append(log)
            return None
        
        fill_price = order_details.get('average_price', order_price)
        investment = fill_price * total_qty
        
        # Calculate target and stop loss prices
        target_price = round_to_tick(fill_price * (1 + target_pct / 100))
        sl_price = round_to_tick(fill_price * (1 - stop_loss_pct / 100))
        
        log = f"{datetime.now().strftime('%H:%M:%S')} | ✅ ENTRY FILLED | {symbol} | Price: ₹{fill_price:.2f} | Qty: {total_qty}"
        st.session_state.trade_logs.append(log)
        
        time.sleep(1)  # Brief pause before GTT
        
        # Place GTT OCO (Target + SL in single order)
        gtt_result = place_gtt_oco(
            kite=kite,
            tradingsymbol=tradingsymbol,
            exchange=kite.EXCHANGE_NFO,
            quantity=total_qty,
            entry_price=fill_price,
            target_price=target_price,
            sl_price=sl_price
        )
        
        gtt_oco_id = None
        if gtt_result['status'] == 'success':
            gtt_oco_id = gtt_result['gtt_id']
            log = f"{datetime.now().strftime('%H:%M:%S')} | ✅ GTT OCO PLACED | {symbol} | ID: {gtt_oco_id} | Target: ₹{target_price:.2f} | SL: ₹{sl_price:.2f}"
            st.session_state.trade_logs.append(log)
        else:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ⚠️ GTT OCO FAILED | {symbol} | {gtt_result['message']}"
            st.session_state.trade_logs.append(log)
        
        # Create trade record
        trade = {
            'id': len(st.session_state.live_positions) + 1,
            'order_id': order_id,
            'gtt_oco_id': gtt_oco_id,  # Single GTT OCO ID (replaces gtt_target_id + gtt_sl_id)
            'gtt_type': 'OCO',
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'contract': tradingsymbol,
            'option_token': option_token,
            'strike': contract['strike'],
            'expiry': contract['expiry'],
            'spot_price': spot_price,
            'entry_price': fill_price,
            'current_price': fill_price,
            'target_price': target_price,
            'sl_price': sl_price,
            'quantity': quantity,
            'lot_size': lot_size,
            'total_qty': total_qty,
            'investment': investment,
            'current_value': investment,
            'pnl': 0,
            'pnl_pct': 0,
            'status': 'OPEN',
            'gtt_status': 'active' if gtt_oco_id else 'failed',
            'confidence': analysis_detail.get('strength', 'MODERATE'),
            'indicators': analysis_detail.get('indicators', [])[:3],
            'max_profit': 0,
            'max_loss': 0
        }
        
        st.session_state.live_positions.append(trade)
        
        log = f"{datetime.now().strftime('%H:%M:%S')} | ✅ LIVE TRADE | {symbol} {signal} | Entry: ₹{fill_price:.2f} | Investment: ₹{investment:.2f} | GTT OCO: {'Active' if gtt_oco_id else 'Failed'}"
        st.session_state.trade_logs.append(log)
        
        return trade
        
    except Exception as e:
        log = f"{datetime.now().strftime('%H:%M:%S')} | ❌ LIVE ERROR | {symbol} | {str(e)}"
        st.session_state.trade_logs.append(log)
        return None


def close_live_position(kite, position, reason):
    """
    Manually close a live position.
    
    UPDATED: Handles GTT OCO cancellation (single ID instead of two).
    """
    try:
        # Cancel GTT OCO if exists
        gtt_oco_id = position.get('gtt_oco_id')
        
        if gtt_oco_id:
            try:
                cancel_result = cancel_gtt_order(kite, gtt_oco_id)
                if cancel_result['success']:
                    log = f"{datetime.now().strftime('%H:%M:%S')} | 🗑️ GTT OCO CANCELLED | {position['symbol']} | ID: {gtt_oco_id}"
                    st.session_state.trade_logs.append(log)
            except:
                pass
        
        # Legacy support: Cancel old-style separate GTTs if they exist
        if position.get('gtt_target_id'):
            try:
                kite.delete_gtt(position['gtt_target_id'])
            except:
                pass
        
        if position.get('gtt_sl_id'):
            try:
                kite.delete_gtt(position['gtt_sl_id'])
            except:
                pass
        
        # Place market sell order
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=position['contract'],
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=position['total_qty'],
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET
        )
        
        position['status'] = 'CLOSED'
        position['exit_time'] = datetime.now()
        position['exit_reason'] = reason
        position['exit_order_id'] = order_id
        position['exit_price'] = position['current_price']
        position['gtt_status'] = 'cancelled'
        
        # Update stats
        st.session_state.performance_stats['total_trades'] += 1
        st.session_state.performance_stats['total_pnl'] += position['pnl']
        
        if position['pnl'] > 0:
            st.session_state.performance_stats['winning_trades'] += 1
        else:
            st.session_state.performance_stats['losing_trades'] += 1
        
        total = st.session_state.performance_stats['total_trades']
        wins = st.session_state.performance_stats['winning_trades']
        st.session_state.performance_stats['win_rate'] = (wins / total * 100) if total > 0 else 0
        
        log = f"{datetime.now().strftime('%H:%M:%S')} | 📤 LIVE CLOSED | {position['symbol']} | P&L: ₹{position['pnl']:.2f} | {reason}"
        st.session_state.trade_logs.append(log)
        
        return True
        
    except Exception as e:
        log = f"{datetime.now().strftime('%H:%M:%S')} | ❌ CLOSE ERROR | {position['symbol']} | {str(e)}"
        st.session_state.trade_logs.append(log)
        return False


def update_live_positions(kite, target_pct, stop_loss_pct, enable_trailing,
                         trailing_activation_pct, trailing_target_increase,
                         trailing_sl_gap_pct, ws_manager=None):
    """
    Update live positions with real-time prices.
    
    UPDATED: 
    - Handles GTT OCO status checking
    - Trailing now modifies GTT OCO (cancel old, place new)
    """
    for position in st.session_state.live_positions[:]:
        if position['status'] != 'OPEN':
            continue
        
        try:
            contract = position['contract']
            option_token = position.get('option_token')
            
            # Check if position still exists in broker
            try:
                positions = kite.positions()
                net_positions = positions['net']
                
                position_exists = False
                for broker_pos in net_positions:
                    if broker_pos['tradingsymbol'] == contract and broker_pos['quantity'] != 0:
                        position_exists = True
                        break
                
                if not position_exists:
                    # Position closed (GTT triggered or manual)
                    position['status'] = 'CLOSED'
                    position['exit_time'] = datetime.now()
                    position['exit_reason'] = 'GTT Triggered / Position Closed'
                    position['exit_price'] = position['current_price']
                    position['gtt_status'] = 'triggered'
                    
                    position['current_value'] = position['exit_price'] * position['total_qty']
                    position['pnl'] = position['current_value'] - position['investment']
                    position['pnl_pct'] = (position['pnl'] / position['investment']) * 100
                    
                    st.session_state.performance_stats['total_trades'] += 1
                    st.session_state.performance_stats['total_pnl'] += position['pnl']
                    
                    if position['pnl'] > 0:
                        st.session_state.performance_stats['winning_trades'] += 1
                    else:
                        st.session_state.performance_stats['losing_trades'] += 1
                    
                    log = f"{datetime.now().strftime('%H:%M:%S')} | 🔔 GTT TRIGGERED | {position['symbol']} | P&L: ₹{position['pnl']:.2f}"
                    st.session_state.trade_logs.append(log)
                    continue
            except:
                pass
            
            # Get current price
            current_price = 0
            
            if ws_manager and ws_manager.is_connected and option_token:
                tick = ws_manager.get_tick(option_token)
                if tick and tick.get('ltp', 0) > 0:
                    current_price = tick['ltp']
            
            if current_price == 0:
                try:
                    ltp_data = kite.ltp(f"NFO:{contract}")
                    current_price = ltp_data[f"NFO:{contract}"]['last_price']
                except:
                    continue
            
            # Update position metrics
            position['current_price'] = current_price
            position['current_value'] = current_price * position['total_qty']
            position['pnl'] = position['current_value'] - position['investment']
            position['pnl_pct'] = (position['pnl'] / position['investment']) * 100
            
            if position['pnl'] > position['max_profit']:
                position['max_profit'] = position['pnl']
            if position['pnl'] < position['max_loss']:
                position['max_loss'] = position['pnl']
            
            # Check GTT status periodically
            gtt_oco_id = position.get('gtt_oco_id')
            if gtt_oco_id:
                gtt_status = get_gtt_status(kite, gtt_oco_id)
                if gtt_status['found']:
                    position['gtt_status'] = gtt_status['status']
                else:
                    position['gtt_status'] = 'unknown'
            
            # Trailing logic (if enabled)
            if enable_trailing and position.get('gtt_oco_id'):
                target_price = position.get('target_price', 0)
                entry_price = position['entry_price']
                
                if target_price > 0:
                    price_to_target = target_price - entry_price
                    threshold_price = entry_price + (price_to_target * (trailing_activation_pct / 100))
                    
                    # If price reached threshold, trail the GTT
                    if current_price >= threshold_price:
                        new_target = round_to_tick(current_price * (1 + trailing_target_increase / 100))
                        new_sl = round_to_tick(current_price * (1 - trailing_sl_gap_pct / 100))
                        
                        # Only trail if new target is higher
                        if new_target > target_price:
                            # Modify GTT OCO (cancel old, place new)
                            modify_result = modify_gtt_oco(
                                kite=kite,
                                old_gtt_id=position['gtt_oco_id'],
                                tradingsymbol=contract,
                                exchange=kite.EXCHANGE_NFO,
                                quantity=position['total_qty'],
                                current_price=current_price,
                                new_target=new_target,
                                new_sl=new_sl
                            )
                            
                            if modify_result['success']:
                                position['gtt_oco_id'] = modify_result['new_gtt_id']
                                position['target_price'] = new_target
                                position['sl_price'] = new_sl
                                position['gtt_status'] = 'active'
                                
                                log = f"{datetime.now().strftime('%H:%M:%S')} | 🚀 TRAILING GTT OCO | {position['symbol']} | New Target: ₹{new_target:.2f} | New SL: ₹{new_sl:.2f}"
                                st.session_state.trade_logs.append(log)
                            else:
                                log = f"{datetime.now().strftime('%H:%M:%S')} | ⚠️ TRAILING FAILED | {position['symbol']} | {modify_result['message']}"
                                st.session_state.trade_logs.append(log)
                                
        except Exception as e:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ⚠️ UPDATE ERROR | {position.get('symbol', 'Unknown')} | {str(e)}"
            st.session_state.trade_logs.append(log)


def sync_gtt_status_all_positions(kite):
    """
    Sync GTT status for all open live positions.
    
    Call this periodically to keep GTT status updated.
    
    Returns:
        dict: Summary of GTT statuses
    """
    summary = {
        'active': 0,
        'triggered': 0,
        'cancelled': 0,
        'unknown': 0,
        'total': 0
    }
    
    try:
        # Get all GTTs from broker
        all_gtts = get_gtt_orders(kite)
        gtt_map = {gtt['id']: gtt for gtt in all_gtts}
        
        for position in st.session_state.live_positions:
            if position['status'] != 'OPEN':
                continue
            
            summary['total'] += 1
            gtt_id = position.get('gtt_oco_id')
            
            if gtt_id and gtt_id in gtt_map:
                status = gtt_map[gtt_id].get('status', 'unknown')
                position['gtt_status'] = status
                
                if status == 'active':
                    summary['active'] += 1
                elif status in ['triggered', 'complete']:
                    summary['triggered'] += 1
                elif status in ['cancelled', 'deleted']:
                    summary['cancelled'] += 1
                else:
                    summary['unknown'] += 1
            else:
                position['gtt_status'] = 'not_found'
                summary['unknown'] += 1
        
        return summary
        
    except Exception as e:
        print(f"Error syncing GTT status: {e}")
        return summary

# ============================================================================
# END OF CHUNK 6 (UPDATED)
# Changes Summary:
# 1. Added place_gtt_oco() - Places single GTT with Target + SL
# 2. Added get_gtt_orders() - Fetches all GTT orders with details
# 3. Added get_gtt_status() - Checks specific GTT status
# 4. Added cancel_gtt_order() - Cancels a GTT
# 5. Added modify_gtt_oco() - Modifies GTT by cancel+replace
# 6. Updated place_live_trade() to use GTT OCO
# 7. Updated close_live_position() to handle OCO cancellation
# 8. Updated update_live_positions() for OCO trailing
# 9. Added sync_gtt_status_all_positions() for bulk status update
# ============================================================================



# ============================================================================
# CHUNK 7: SCANNING AND TRADING LOGIC (UPDATED)
# Description: Main scanning and trading loop with expiry day filter AND daily limits
# Changes Made:
#   - ADDED: Daily profit/loss limit check before scanning
#   - ADDED: Auto-stop trading when limits hit
#   - ADDED: Close all positions option when daily SL hit
#   - ADDED: Congratulations message when target reached
# Dependencies: Chunk 1-6
# ============================================================================

def subscribe_to_filtered_stocks(ws_manager, signals):
    """
    Subscribe WebSocket to tokens of stocks with active signals.
    """
    if not ws_manager or not ws_manager.is_connected:
        return
    
    tokens_to_subscribe = []
    
    for sig in signals:
        token = sig.get('stock_info', {}).get('token')
        if token:
            tokens_to_subscribe.append(token)
        
        option_token = sig.get('option_token')
        if option_token:
            tokens_to_subscribe.append(option_token)
    
    if tokens_to_subscribe:
        ws_manager.subscribe_tokens(tokens_to_subscribe, mode='full')


def subscribe_to_open_positions(ws_manager, kite, paper_mode):
    """
    Subscribe WebSocket to tokens of open positions for real-time P&L tracking.
    """
    if not ws_manager or not ws_manager.is_connected:
        return
    
    positions = st.session_state.paper_positions if paper_mode else st.session_state.live_positions
    open_positions = [p for p in positions if p['status'] == 'OPEN']
    
    tokens = []
    for pos in open_positions:
        option_token = pos.get('option_token')
        if option_token:
            tokens.append(option_token)
    
    if tokens:
        ws_manager.subscribe_tokens(tokens, mode='full')


def get_available_capital(kite, paper_mode, max_capital_risk_pct):
    """Calculate available capital considering open positions."""
    if paper_mode:
        return st.session_state.paper_capital * (max_capital_risk_pct / 100)
    else:
        try:
            margins = kite.margins()
            total_balance = margins['equity']['available']['live_balance']
            
            open_positions = [p for p in st.session_state.live_positions if p['status'] == 'OPEN']
            locked_capital = sum(p['investment'] for p in open_positions)
            
            available = total_balance - locked_capital
            return max(0, available * (max_capital_risk_pct / 100))
        except:
            return 0


def check_global_expiry_day():
    """
    Check if today is expiry day for any major index.
    
    Returns:
        dict: {is_expiry_day: bool, indices: list of expiring indices}
    """
    today = datetime.now().date()
    weekday = today.weekday()  # 0=Monday, 6=Sunday
    
    expiring_indices = []
    
    # Standard weekly expiry days (approximate)
    # MIDCPNIFTY: Monday
    # FINNIFTY: Tuesday
    # BANKNIFTY: Wednesday
    # NIFTY: Thursday
    
    if weekday == 0:
        expiring_indices.append('MIDCPNIFTY')
    elif weekday == 1:
        expiring_indices.append('FINNIFTY')
    elif weekday == 2:
        expiring_indices.append('BANKNIFTY')
    elif weekday == 3:
        expiring_indices.append('NIFTY')
        # Also check if last Thursday of month (stock options expiry)
        # This is simplified - in production, check actual expiry calendar
    
    return {
        'is_expiry_day': len(expiring_indices) > 0,
        'expiring_indices': expiring_indices,
        'weekday': weekday,
        'date': today
    }


def close_all_open_positions(kite, reason, ws_manager=None):
    """
    Close all open positions (for daily stop loss scenario).
    
    Args:
        kite: KiteConnect instance
        reason: Reason for closing all positions
        ws_manager: WebSocketManager instance
        
    Returns:
        int: Number of positions closed
    """
    positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
    open_positions = [p for p in positions if p['status'] == 'OPEN']
    
    closed_count = 0
    
    for position in open_positions:
        try:
            if st.session_state.paper_mode:
                # Close paper position
                close_paper_position(position, reason)
                closed_count += 1
            else:
                # Close live position
                if close_live_position(kite, position, reason):
                    closed_count += 1
        except Exception as e:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ERROR | Failed to close {position['symbol']}: {e}"
            st.session_state.trade_logs.append(log)
    
    return closed_count


def scan_and_trade(kite, option_stocks, target_pct, stop_loss_pct, max_capital_risk_pct, 
                   max_concurrent_trades, enable_trailing, trailing_activation_pct, 
                   trailing_target_increase, trailing_sl_gap_pct, ws_manager=None,
                   avoid_expiry_day=True, close_all_on_daily_sl=False):  # NEW PARAMETER
    """
    Main scanning and trading function.
    
    UPDATED: Now includes daily profit/loss limit checking.
    
    Flow:
    1. Check kill switch and market hours
    2. CHECK DAILY PROFIT/LOSS LIMITS (NEW)
    3. Update existing positions
    4. Check position limits
    5. Check expiry day
    6. Scan for new signals (skip expiry day contracts)
    7. Execute trades based on signals
    
    Args:
        ... (existing args)
        close_all_on_daily_sl: If True, close all positions when daily SL hit
    """
    
    # Kill switch check
    if st.session_state.kill_switch:
        return
    
    # Already stopped for daily limit
    if st.session_state.daily_target_reached or st.session_state.daily_sl_reached:
        return
    
    # Market hours check
    current_time = datetime.now().time()
    if current_time < MARKET_OPEN.time() or current_time >= STOP_TRADING.time():
        return
    
    # === NEW: CHECK DAILY PROFIT/LOSS LIMITS ===
    should_stop, stop_reason, pnl_info = check_daily_limits()
    
    if should_stop:
        if stop_reason == 'PROFIT_TARGET':
            st.session_state.daily_target_reached = True
            st.session_state.trading_active = False
            st.session_state.trading_stopped_reason = 'PROFIT_TARGET'
            
            log = f"{datetime.now().strftime('%H:%M:%S')} | 🎉 DAILY TARGET REACHED! | P&L: ₹{pnl_info['total_pnl']:,.2f} | Target: ₹{st.session_state.daily_profit_target:,.2f}"
            st.session_state.trade_logs.append(log)
            st.session_state.trade_logs.append(f"{datetime.now().strftime('%H:%M:%S')} | 🛑 TRADING STOPPED | Congratulations! You've reached your daily profit target!")
            return
        
        elif stop_reason == 'DAILY_STOP_LOSS':
            st.session_state.daily_sl_reached = True
            st.session_state.trading_active = False
            st.session_state.trading_stopped_reason = 'DAILY_STOP_LOSS'
            
            log = f"{datetime.now().strftime('%H:%M:%S')} | 🛑 DAILY STOP LOSS HIT! | P&L: ₹{pnl_info['total_pnl']:,.2f} | Max Loss: -₹{st.session_state.daily_stop_loss:,.2f}"
            st.session_state.trade_logs.append(log)
            
            # Optionally close all positions
            if close_all_on_daily_sl:
                closed_count = close_all_open_positions(kite, "DAILY STOP LOSS - Emergency Exit", ws_manager)
                log = f"{datetime.now().strftime('%H:%M:%S')} | 🚨 EMERGENCY EXIT | Closed {closed_count} positions"
                st.session_state.trade_logs.append(log)
            
            st.session_state.trade_logs.append(f"{datetime.now().strftime('%H:%M:%S')} | 🛑 TRADING STOPPED | Daily loss limit reached. Take a break.")
            return
    
    # === GLOBAL EXPIRY DAY CHECK ===
    expiry_info = check_global_expiry_day()
    if expiry_info['is_expiry_day'] and avoid_expiry_day:
        log = f"{datetime.now().strftime('%H:%M:%S')} | ⚠️ EXPIRY DAY | {', '.join(expiry_info['expiring_indices'])} expiring today - Extra caution applied"
        if log not in st.session_state.trade_logs[-10:]:
            st.session_state.trade_logs.append(log)
    
    # Update existing positions
    if st.session_state.paper_mode:
        update_paper_positions(kite, target_pct, stop_loss_pct, ws_manager)
    else:
        update_live_positions(kite, target_pct, stop_loss_pct, enable_trailing,
                             trailing_activation_pct, trailing_target_increase,
                             trailing_sl_gap_pct, ws_manager)
    
    # === RE-CHECK LIMITS AFTER POSITION UPDATE (in case a trade closed and hit limits) ===
    should_stop, stop_reason, pnl_info = check_daily_limits()
    if should_stop:
        if stop_reason == 'PROFIT_TARGET':
            st.session_state.daily_target_reached = True
            st.session_state.trading_active = False
            st.session_state.trading_stopped_reason = 'PROFIT_TARGET'
            log = f"{datetime.now().strftime('%H:%M:%S')} | 🎉 DAILY TARGET REACHED AFTER TRADE CLOSE! | P&L: ₹{pnl_info['total_pnl']:,.2f}"
            st.session_state.trade_logs.append(log)
            return
        elif stop_reason == 'DAILY_STOP_LOSS':
            st.session_state.daily_sl_reached = True
            st.session_state.trading_active = False
            st.session_state.trading_stopped_reason = 'DAILY_STOP_LOSS'
            log = f"{datetime.now().strftime('%H:%M:%S')} | 🛑 DAILY STOP LOSS HIT AFTER TRADE CLOSE! | P&L: ₹{pnl_info['total_pnl']:,.2f}"
            st.session_state.trade_logs.append(log)
            return
    
    # Subscribe to open positions
    subscribe_to_open_positions(ws_manager, kite, st.session_state.paper_mode)
    
    # Check position limits
    positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
    open_positions_count = len([p for p in positions if p['status'] == 'OPEN'])
    
    if open_positions_count >= max_concurrent_trades:
        log = f"{datetime.now().strftime('%H:%M:%S')} | ⏳ WAITING | {open_positions_count}/{max_concurrent_trades} positions open"
        if log not in st.session_state.trade_logs[-5:]:
            st.session_state.trade_logs.append(log)
        return
    
    # Calculate available capital
    available_capital = get_available_capital(kite, st.session_state.paper_mode, max_capital_risk_pct)
    
    st.session_state.active_signals = []
    all_signals = []
    skipped_expiry_count = 0  # Track skipped due to expiry
    
    INDEX_SYMBOLS = ['NIFTY']
    
    # Filter stocks based on index_only_mode
    if st.session_state.index_only_mode:
        stocks_to_scan = [s for s in option_stocks if s['symbol'] in INDEX_SYMBOLS]
        if not stocks_to_scan:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ⚠️ INDEX MODE | No indices in stock list"
            st.session_state.trade_logs.append(log)
            return
    else:
        stocks_to_scan = option_stocks
    
    # Scan stocks (filtered based on mode)
    for stock_info in stocks_to_scan:
        if st.session_state.kill_switch:
            break
        
        # Re-check daily limits during scan (in case positions close mid-scan)
        should_stop, _, _ = check_daily_limits()
        if should_stop:
            break
        
        try:
            symbol = stock_info['symbol']
            token = stock_info['token']
            
            # Skip if already have position
            existing = [p for p in positions if p['symbol'] == symbol and p['status'] == 'OPEN']
            if existing:
                continue
            
            # === SKIP INDEX ON ITS EXPIRY DAY ===
            if avoid_expiry_day and symbol in expiry_info.get('expiring_indices', []):
                skipped_expiry_count += 1
                continue
            
            # Get spot price
            spot_price = get_real_spot_price(kite, symbol, token, ws_manager)
            
            if spot_price <= 0:
                continue
            
            # Run analysis
            analyzer = ProfessionalTradingAnalysis(kite, symbol, token, ws_manager)
            df = analyzer.get_historical_data()
            
            if df is None or len(df) < 200:
                continue
            
            signal, confidence, analysis = analyzer.generate_professional_signal(df)
            
            if signal and confidence >= 60:
                option_type = 'CE' if signal == 'CALL' else 'PE'
                
                # Get option price with expiry day filtering
                option_data = get_real_option_price(
                    kite, symbol, option_type, spot_price, ws_manager,
                    avoid_expiry_day=avoid_expiry_day
                )
                
                if option_data:
                    days_to_expiry = option_data.get('days_to_expiry', 99)
                    is_expiry_day = option_data.get('is_expiry_day', False)
                    
                    # Skip if expiry day contract somehow got through
                    if is_expiry_day and avoid_expiry_day:
                        skipped_expiry_count += 1
                        continue
                    
                    # Add warning for low DTE
                    expiry_note = ''
                    if days_to_expiry <= 1:
                        expiry_note = '⚠️ 0-1 DTE'
                    elif days_to_expiry <= 2:
                        expiry_note = '⚡ 2 DTE'
                    
                    all_signals.append({
                        'stock_info': stock_info,
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'spot_price': spot_price,
                        'option_premium': option_data['premium'],
                        'option_contract': option_data['contract'],
                        'option_token': option_data['token'],
                        'strike': option_data['strike'],
                        'expiry': option_data['expiry'],
                        'lot_size': stock_info.get('lot_size', 1),
                        'strength': analysis.get('strength', 'MODERATE'),
                        'indicators': analysis.get('indicators', [])[:3],
                        'timestamp': datetime.now(),
                        'analysis': analysis,
                        'buy_score': analysis.get('buy_score', 0),
                        'sell_score': analysis.get('sell_score', 0),
                        'price_source': option_data['source'],
                        'days_to_expiry': days_to_expiry,
                        'is_expiry_day': is_expiry_day,
                        'expiry_note': expiry_note,
                        'category_count': analysis.get('category_count', 0),
                        'confirmation_count': analysis.get('confirmation_count', 0)
                    })
        except Exception as e:
            pass
    
    # Log skipped expiry trades
    if skipped_expiry_count > 0:
        log = f"{datetime.now().strftime('%H:%M:%S')} | 🚫 EXPIRY FILTER | Skipped {skipped_expiry_count} signals due to expiry day"
        st.session_state.trade_logs.append(log)
    
    # Sort by confidence and score
    all_signals.sort(key=lambda x: (x['confidence'], max(x['buy_score'], x['sell_score'])), reverse=True)
    
    # Store all signals
    st.session_state.active_signals = all_signals
    
    # Subscribe to filtered stocks
    subscribe_to_filtered_stocks(ws_manager, all_signals[:20])
    
    # Calculate trades to take
    trades_to_take = max_concurrent_trades - open_positions_count
    best_signals = all_signals[:trades_to_take]
    
    # Execute trades
    for sig in best_signals:
        # Check daily limits before each trade
        should_stop, _, _ = check_daily_limits()
        if should_stop:
            break
        
        try:
            symbol = sig['symbol']
            signal = sig['signal']
            spot_price = sig['spot_price']
            stock_info = sig['stock_info']
            analysis = sig['analysis']
            lot_size = sig['lot_size']
            option_premium = sig['option_premium']
            days_to_expiry = sig.get('days_to_expiry', 99)
            
            # Final expiry day check
            if sig.get('is_expiry_day', False) and avoid_expiry_day:
                log = f"{datetime.now().strftime('%H:%M:%S')} | 🚫 SKIPPED | {symbol} | Expiry day - not trading"
                st.session_state.trade_logs.append(log)
                continue
            
            if st.session_state.paper_mode:
                cost_per_lot = option_premium * lot_size
                
                if cost_per_lot > 0:
                    max_lots = int(available_capital / cost_per_lot)
                    num_lots = max(1, min(max_lots, 2))
                    
                    if cost_per_lot * num_lots <= available_capital:
                        trade = place_paper_trade(kite, symbol, signal, spot_price, num_lots, 
                                         stock_info, analysis, ws_manager)
                        
                        # Add expiry info to trade record
                        if trade:
                            trade['days_to_expiry'] = days_to_expiry
                            trade['expiry_note'] = sig.get('expiry_note', '')
            else:
                trade = place_live_trade(kite, symbol, signal, spot_price, 1, stock_info, 
                                analysis, target_pct, stop_loss_pct, ws_manager)
                
                if trade:
                    trade['days_to_expiry'] = days_to_expiry
                    trade['expiry_note'] = sig.get('expiry_note', '')
        
        except Exception as e:
            log = f"{datetime.now().strftime('%H:%M:%S')} | ERROR | {sig['symbol']} - {str(e)}"
            st.session_state.trade_logs.append(log)


def update_live_positions(kite, target_pct, stop_loss_pct, enable_trailing,
                         trailing_activation_pct, trailing_target_increase,
                         trailing_sl_gap_pct, ws_manager=None):
    """Update live positions with real-time prices."""
    for position in st.session_state.live_positions[:]:
        if position['status'] != 'OPEN':
            continue
        
        try:
            contract = position['contract']
            option_token = position.get('option_token')
            
            # Check if position still exists
            try:
                positions = kite.positions()
                net_positions = positions['net']
                
                position_exists = False
                for broker_pos in net_positions:
                    if broker_pos['tradingsymbol'] == contract and broker_pos['quantity'] != 0:
                        position_exists = True
                        break
                
                if not position_exists:
                    position['status'] = 'CLOSED'
                    position['exit_time'] = datetime.now()
                    position['exit_reason'] = 'Position Closed'
                    position['exit_price'] = position['current_price']
                    
                    position['current_value'] = position['exit_price'] * position['total_qty']
                    position['pnl'] = position['current_value'] - position['investment']
                    position['pnl_pct'] = (position['pnl'] / position['investment']) * 100
                    
                    st.session_state.performance_stats['total_trades'] += 1
                    st.session_state.performance_stats['total_pnl'] += position['pnl']
                    
                    if position['pnl'] > 0:
                        st.session_state.performance_stats['winning_trades'] += 1
                    else:
                        st.session_state.performance_stats['losing_trades'] += 1
                    
                    log = f"{datetime.now().strftime('%H:%M:%S')} | 🔔 CLOSED | {position['symbol']} | P&L: ₹{position['pnl']:.2f}"
                    st.session_state.trade_logs.append(log)
                    continue
            except:
                pass
            
            # Get current price
            current_price = 0
            
            if ws_manager and ws_manager.is_connected and option_token:
                tick = ws_manager.get_tick(option_token)
                if tick and tick.get('ltp', 0) > 0:
                    current_price = tick['ltp']
            
            if current_price == 0:
                try:
                    ltp_data = kite.ltp(f"NFO:{contract}")
                    current_price = ltp_data[f"NFO:{contract}"]['last_price']
                except:
                    continue
            
            position['current_price'] = current_price
            position['current_value'] = current_price * position['total_qty']
            position['pnl'] = position['current_value'] - position['investment']
            position['pnl_pct'] = (position['pnl'] / position['investment']) * 100
            
            if position['pnl'] > position['max_profit']:
                position['max_profit'] = position['pnl']
            if position['pnl'] < position['max_loss']:
                position['max_loss'] = position['pnl']
            
            # Trailing logic
            if enable_trailing:
                target_price = position.get('target_price', 0)
                entry_price = position['entry_price']
                
                if target_price > 0:
                    price_to_target = target_price - entry_price
                    threshold_price = entry_price + (price_to_target * (trailing_activation_pct / 100))
                    
                    if current_price >= threshold_price:
                        new_target = round_to_tick(current_price * (1 + trailing_target_increase / 100))
                        new_sl = round_to_tick(new_target * (1 - trailing_sl_gap_pct / 100))
                        
                        if new_target > target_price:
                            try:
                                if position.get('gtt_target_id'):
                                    kite.delete_gtt(position['gtt_target_id'])
                                if position.get('gtt_sl_id'):
                                    kite.delete_gtt(position['gtt_sl_id'])
                                
                                time.sleep(1)
                                
                                new_gtt_target = kite.place_gtt(
                                    trigger_type=kite.GTT_TYPE_SINGLE,
                                    tradingsymbol=contract,
                                    exchange=kite.EXCHANGE_NFO,
                                    trigger_values=[new_target],
                                    last_price=current_price,
                                    orders=[{
                                        'exchange': kite.EXCHANGE_NFO,
                                        'tradingsymbol': contract,
                                        'transaction_type': kite.TRANSACTION_TYPE_SELL,
                                        'quantity': position['total_qty'],
                                        'order_type': kite.ORDER_TYPE_LIMIT,
                                        'product': kite.PRODUCT_MIS,
                                        'price': new_target
                                    }]
                                )
                                
                                new_gtt_sl = kite.place_gtt(
                                    trigger_type=kite.GTT_TYPE_SINGLE,
                                    tradingsymbol=contract,
                                    exchange=kite.EXCHANGE_NFO,
                                    trigger_values=[new_sl],
                                    last_price=current_price,
                                    orders=[{
                                        'exchange': kite.EXCHANGE_NFO,
                                        'tradingsymbol': contract,
                                        'transaction_type': kite.TRANSACTION_TYPE_SELL,
                                        'quantity': position['total_qty'],
                                        'order_type': kite.ORDER_TYPE_LIMIT,
                                        'product': kite.PRODUCT_MIS,
                                        'price': new_sl
                                    }]
                                )
                                
                                position['gtt_target_id'] = new_gtt_target
                                position['gtt_sl_id'] = new_gtt_sl
                                position['target_price'] = new_target
                                position['sl_price'] = new_sl
                                
                                log = f"{datetime.now().strftime('%H:%M:%S')} | 🚀 TRAILING | {position['symbol']} | Target: ₹{new_target:.2f} | SL: ₹{new_sl:.2f}"
                                st.session_state.trade_logs.append(log)
                            except:
                                pass
        except:
            pass

# ============================================================================
# END OF CHUNK 7 (UPDATED)
# Changes Summary:
# 1. Added close_all_on_daily_sl parameter to scan_and_trade()
# 2. Added daily limit check at start of scan_and_trade()
# 3. Added daily limit re-check after position updates
# 4. Added daily limit check during scan loop
# 5. Added daily limit check before each trade execution
# 6. Added close_all_open_positions() function
# 7. Proper logging for daily target/SL events
# 8. Auto-stop trading when limits hit
# ============================================================================

# ============================================================================
# CHUNK 8: MAIN APPLICATION UI - PART 1 (SIDEBAR) - UPDATED FOR OPTION 1
# Description: Streamlit UI for login, controls, and sidebar
# Changes Made:
#   - COMPLETELY REDESIGNED: render_login_page() with 2-step process
#   - Step 1: Enter API Key and API Secret
#   - Step 2: Generate URL, paste token, complete login
#   - ADDED: Clear credentials button
#   - UPDATED: WebSocket connection uses st.session_state.api_key
# Dependencies: Chunk 1-7
# ============================================================================

def render_login_page():
    """
    Render the login page for Zerodha authentication.
    
    UPDATED FOR OPTION 1:
    - Two-step login process
    - Step 1: Enter API credentials (stored in session state)
    - Step 2: Generate login URL, paste request token, complete login
    """
    st.header("🔐 Zerodha Login")
    
    # =========================================================================
    # STEP 1: API CREDENTIALS
    # =========================================================================
    st.markdown("### Step 1: Enter API Credentials")
    st.caption("Your credentials are stored only in this session and never saved permanently.")
    
    # Check if credentials are already saved
    if st.session_state.credentials_saved:
        # Show saved credentials (masked)
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.success(f"✅ API Key: {st.session_state.api_key[:4]}...{st.session_state.api_key[-4:]}")
        
        with col2:
            st.success("✅ API Secret: ••••••••••••")
        
        with col3:
            if st.button("🔄 Change", use_container_width=True):
                clear_api_credentials()
                st.rerun()
    
    else:
        # Input form for credentials
        with st.form("credentials_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                api_key_input = st.text_input(
                    "API Key",
                    placeholder="Enter your Zerodha API Key",
                    help="Found in Kite Connect developer console"
                )
            
            with col2:
                api_secret_input = st.text_input(
                    "API Secret",
                    type="password",
                    placeholder="Enter your Zerodha API Secret",
                    help="Found in Kite Connect developer console"
                )
            
            submit_credentials = st.form_submit_button("💾 Save Credentials", use_container_width=True)
            
            if submit_credentials:
                # Validate credentials
                is_valid, error_msg = validate_api_credentials(api_key_input, api_secret_input)
                
                if is_valid:
                    # Save to session state
                    st.session_state.api_key = api_key_input.strip()
                    st.session_state.api_secret = api_secret_input.strip()
                    st.session_state.credentials_saved = True
                    st.success("✅ Credentials saved successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {error_msg}")
        
        # Show help
        with st.expander("ℹ️ Where to find API credentials?"):
            st.markdown("""
            1. Go to [Kite Connect Developer Console](https://developers.kite.trade/)
            2. Login with your Zerodha credentials
            3. Create a new app or use existing one
            4. Copy the **API Key** and **API Secret**
            
            **Note:** Make sure your app's redirect URL is set correctly.
            """)
        
        # Stop here if credentials not saved
        return
    
    st.markdown("---")
    
    # =========================================================================
    # STEP 2: LOGIN & TOKEN EXCHANGE
    # =========================================================================
    st.markdown("### Step 2: Login & Get Token")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**A. Generate Login URL**")
        
        if st.button("🔗 Generate Login URL", use_container_width=True):
            try:
                # Initialize Kite with user's API key
                kite = KiteConnect(api_key=st.session_state.api_key)
                login_url = kite.login_url()
                st.session_state.kite = kite
                st.session_state.login_url = login_url
                st.success("✅ Login URL generated!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        # Show login URL if generated
        if hasattr(st.session_state, 'login_url') and st.session_state.login_url:
            st.code(st.session_state.login_url, language=None)
            st.caption("👆 Click the URL above, login to Zerodha, then copy the `request_token` from the redirect URL")
            
            # Instructions
            with st.expander("📋 How to get request_token?"):
                st.markdown("""
                1. Click the login URL above
                2. Login with your Zerodha credentials
                3. Complete 2FA (TOTP/PIN)
                4. You'll be redirected to a URL like:
                   ```
                   https://your-redirect-url.com/?request_token=XXXXXX&action=login&status=success
                   ```
                5. Copy the value after `request_token=` (before `&`)
                6. Paste it in the field on the right
                """)
    
    with col2:
        st.markdown("**B. Complete Login**")
        
        request_token = st.text_input(
            "Request Token",
            type="password",
            placeholder="Paste request_token here",
            help="Copy from the redirect URL after logging in"
        )
        
        if st.button("✅ Complete Login", use_container_width=True, type="primary"):
            if not request_token:
                st.error("❌ Please enter the request token")
            else:
                try:
                    # Initialize Kite if not already
                    if st.session_state.kite is None:
                        st.session_state.kite = KiteConnect(api_key=st.session_state.api_key)
                    
                    # Generate session using user's API secret
                    data = st.session_state.kite.generate_session(
                        request_token.strip(), 
                        api_secret=st.session_state.api_secret
                    )
                    
                    # Save access token
                    st.session_state.access_token = data["access_token"]
                    st.session_state.kite.set_access_token(st.session_state.access_token)
                    
                    # Clear login URL
                    if hasattr(st.session_state, 'login_url'):
                        del st.session_state.login_url
                    
                    st.success("✅ Login Successful!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Login Failed: {e}")
                    st.caption("Common issues: Token expired, wrong credentials, or token already used")
    
    # =========================================================================
    # LOGOUT OPTION (if already logged in but want to re-login)
    # =========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔓 Clear All & Start Fresh", use_container_width=True):
            clear_api_credentials()
            if hasattr(st.session_state, 'login_url'):
                del st.session_state.login_url
            st.rerun()


def render_sidebar():
    """Render the sidebar with controls and settings."""
    
    # === KILL SWITCH ===
    st.sidebar.markdown("### 🚨 KILL SWITCH")
    if st.sidebar.button(
        "🛑 STOP ALL" if not st.session_state.kill_switch else "▶️ RESUME",
        type="primary",
        use_container_width=True
    ):
        st.session_state.kill_switch = not st.session_state.kill_switch
        if st.session_state.kill_switch:
            st.session_state.trading_active = False
        st.rerun()
    
    if st.session_state.kill_switch:
        st.sidebar.error("🛑 ALL TRADING STOPPED")
    else:
        st.sidebar.success("✅ System Active")
    
    st.sidebar.markdown("---")
    
    # === LOGGED IN USER INFO (NEW) ===
    if st.session_state.credentials_saved and st.session_state.access_token:
        st.sidebar.markdown("### 👤 Session")
        st.sidebar.success(f"API: {st.session_state.api_key[:4]}...{st.session_state.api_key[-4:]}")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            clear_api_credentials()
            st.rerun()
        st.sidebar.markdown("---")
    
    # === DAILY TARGET/SL STATUS ===
    pnl_info = calculate_daily_pnl()
    total_pnl = pnl_info['total_pnl']
    
    if st.session_state.daily_target_reached:
        st.sidebar.success("🎉 DAILY TARGET REACHED!")
        st.sidebar.balloons()
    elif st.session_state.daily_sl_reached:
        st.sidebar.error("🛑 DAILY STOP LOSS HIT!")
    
    # === EXPIRY DAY WARNING ===
    expiry_info = check_global_expiry_day()
    if expiry_info['is_expiry_day']:
        st.sidebar.warning(f"⚠️ EXPIRY DAY: {', '.join(expiry_info['expiring_indices'])}")
    
    # === WEBSOCKET STATUS (UPDATED - uses session state credentials) ===
    st.sidebar.markdown("### 📡 WebSocket")
    ws_manager = st.session_state.ws_manager
    
    if ws_manager and ws_manager.is_connected:
        st.sidebar.success(f"🟢 Connected ({ws_manager.get_subscribed_count()} tokens)")
    else:
        st.sidebar.warning("🟡 Not Connected")
        
        if st.sidebar.button("🔌 Connect WebSocket", use_container_width=True):
            # Check if we have credentials
            if not st.session_state.api_key:
                st.sidebar.error("❌ No API Key found. Please login first.")
            elif not st.session_state.access_token:
                st.sidebar.error("❌ No access token. Please complete login first.")
            else:
                with st.spinner("Connecting..."):
                    # Use credentials from session state
                    ws_manager = WebSocketManager(
                        st.session_state.api_key,  # From session state, not hardcoded
                        st.session_state.access_token
                    )
                    if ws_manager.connect():
                        st.session_state.ws_manager = ws_manager
                        st.success("✅ WebSocket Connected!")
                        st.rerun()
                    else:
                        st.error("❌ Connection Failed")
    
    st.sidebar.markdown("---")
    
    # === TRADING MODE ===
    st.sidebar.markdown("### 📊 Trading Mode")
    mode_col1, mode_col2 = st.sidebar.columns(2)
    
    with mode_col1:
        if st.button("📄 Paper",
                    type="primary" if st.session_state.paper_mode else "secondary",
                    use_container_width=True):
            st.session_state.paper_mode = True
            st.rerun()
    
    with mode_col2:
        if st.button("💰 Live",
                    type="primary" if not st.session_state.paper_mode else "secondary",
                    use_container_width=True):
            st.session_state.paper_mode = False
            st.rerun()
    
    if st.session_state.paper_mode:
        st.sidebar.info("📄 PAPER TRADING MODE")
        
        custom_capital = st.sidebar.number_input(
            "Starting Capital",
            min_value=10000,
            max_value=10000000,
            value=int(st.session_state.initial_paper_capital),
            step=10000
        )
        
        if custom_capital != st.session_state.initial_paper_capital:
            st.session_state.initial_paper_capital = custom_capital
            st.session_state.paper_capital = custom_capital
            st.rerun()
        
        st.sidebar.metric("Available Capital", f"₹{st.session_state.paper_capital:,.0f}")
    else:
        st.sidebar.warning("💰 LIVE TRADING MODE")
        try:
            margins = st.session_state.kite.margins()
            available = margins['equity']['available']['live_balance']
            st.sidebar.metric("Live Balance", f"₹{available:,.2f}")
        except:
            st.sidebar.metric("Live Balance", "Unable to fetch")
    
    st.sidebar.markdown("---")
    
    # === DAILY PROFIT/LOSS LIMITS ===
    st.sidebar.markdown("### 🎯 Daily Profit/Loss Limits")
    
    pnl_color = "🟢" if total_pnl >= 0 else "🔴"
    st.sidebar.metric(
        "Today's P&L", 
        f"{pnl_color} ₹{total_pnl:,.2f}",
        delta=f"Realized: ₹{pnl_info['realized_pnl']:,.2f}"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.caption(f"Realized: ₹{pnl_info['realized_pnl']:,.2f}")
    with col2:
        st.caption(f"Unrealized: ₹{pnl_info['unrealized_pnl']:,.2f}")
    
    daily_profit_target = st.sidebar.number_input(
        "🎯 Daily Profit Target (₹)",
        min_value=0,
        max_value=100000,
        value=int(st.session_state.daily_profit_target),
        step=500,
        help="Trading stops when total P&L reaches this amount. Set to 0 to disable."
    )
    
    if daily_profit_target != st.session_state.daily_profit_target:
        st.session_state.daily_profit_target = daily_profit_target
    
    daily_stop_loss = st.sidebar.number_input(
        "🛑 Daily Stop Loss (₹)",
        min_value=0,
        max_value=50000,
        value=int(st.session_state.daily_stop_loss),
        step=500,
        help="Trading stops when total loss reaches this amount. Set to 0 to disable."
    )
    
    if daily_stop_loss != st.session_state.daily_stop_loss:
        st.session_state.daily_stop_loss = daily_stop_loss
    
    # Progress to target/SL
    if daily_profit_target > 0 or daily_stop_loss > 0:
        st.sidebar.markdown("**Progress:**")
        
        if daily_profit_target > 0 and total_pnl >= 0:
            progress_to_target = min(total_pnl / daily_profit_target, 1.0)
            st.sidebar.progress(progress_to_target)
            st.sidebar.caption(f"🎯 {progress_to_target * 100:.1f}% to target (₹{daily_profit_target:,.0f})")
        
        if daily_stop_loss > 0 and total_pnl < 0:
            progress_to_sl = min(abs(total_pnl) / daily_stop_loss, 1.0)
            st.sidebar.progress(progress_to_sl)
            st.sidebar.caption(f"🛑 {progress_to_sl * 100:.1f}% to stop loss (₹{daily_stop_loss:,.0f})")
    
    close_all_on_daily_sl = st.sidebar.checkbox(
        "🚨 Close all positions on Daily SL",
        value=False,
        help="If enabled, all open positions will be closed immediately when daily stop loss is hit"
    )
    
    if st.session_state.daily_target_reached or st.session_state.daily_sl_reached:
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Reset Daily Limits", use_container_width=True, type="primary"):
            reset_daily_limits()
            st.session_state.trading_stopped_reason = None
            st.rerun()
        st.sidebar.caption("⚠️ Click to resume trading (use with caution)")
    
    st.sidebar.markdown("---")
    
    # === PERFORMANCE STATS ===
    st.sidebar.markdown("### 📈 Performance")
    stats = st.session_state.performance_stats
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Trades", stats['total_trades'])
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col2:
        st.metric("Total P&L", f"₹{stats['total_pnl']:.0f}")
        st.metric("Best Trade", f"₹{stats['best_trade']:.0f}")
    
    st.sidebar.markdown("---")
    
    # === RISK MANAGEMENT ===
    st.sidebar.markdown("### 🛡️ Risk Management")
    
    target_pct = st.sidebar.slider(
        "Target %", 
        min_value=1, 
        max_value=100, 
        value=20,
        help="Recommended: 20-25% for 2:1 R:R ratio"
    )
    
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss %", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Recommended: 10% for 2:1 R:R ratio"
    )
    
    rr_ratio = target_pct / stop_loss_pct if stop_loss_pct > 0 else 0
    if rr_ratio >= 2:
        rr_color = "🟢"
        rr_status = "Excellent"
    elif rr_ratio >= 1.5:
        rr_color = "🟡"
        rr_status = "Acceptable"
    else:
        rr_color = "🔴"
        rr_status = "Poor"
    
    st.sidebar.markdown(f"{rr_color} **R:R Ratio: {rr_ratio:.1f}:1** ({rr_status})")
    
    if rr_ratio < 1.5:
        st.sidebar.error("⚠️ Poor R:R! Increase target or reduce stop loss")
    elif rr_ratio < 2:
        st.sidebar.warning("💡 Consider: Target 20%+ with SL 10%")
    
    max_capital_risk = st.sidebar.slider("Max Capital %", 5, 95, 50)
    
    st.sidebar.markdown("---")
    
    # === EXPIRY DAY SETTINGS ===
    st.sidebar.markdown("### 📅 Expiry Day Settings")
    
    avoid_expiry_day = st.sidebar.checkbox(
        "🚫 Avoid Expiry Day Trading",
        value=True,
        help="Skip trading on expiry day to avoid extreme theta decay and gamma risk"
    )
    
    if avoid_expiry_day:
        st.sidebar.success("✅ Expiry day protection ON")
    else:
        st.sidebar.error("⚠️ WARNING: Expiry day trading enabled - HIGH RISK!")
    
    if expiry_info['is_expiry_day']:
        st.sidebar.info(f"📅 Today's expiries: {', '.join(expiry_info['expiring_indices'])}")
        if avoid_expiry_day:
            st.sidebar.caption("↳ These will be skipped automatically")
    else:
        st.sidebar.caption("No index expiry today")
    
    st.sidebar.markdown("---")
    
    # === INDEX ONLY MODE ===
    st.sidebar.markdown("### 🎯 Trading Universe")
    
    index_only_mode = st.sidebar.checkbox(
        "📊 Index Only Mode",
        value=st.session_state.index_only_mode,
        help="Trade only NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY - Skip all stock options"
    )
    
    if index_only_mode != st.session_state.index_only_mode:
        st.session_state.index_only_mode = index_only_mode
        st.rerun()
    
    if index_only_mode:
        st.sidebar.info("🎯 Trading: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY only")
        st.sidebar.caption("All stock options will be skipped")
    else:
        st.sidebar.caption(f"Trading all {len(st.session_state.option_stocks)} F&O stocks + indices")
    
    st.sidebar.markdown("---")
    
    # === TRAILING SETTINGS ===
    st.sidebar.markdown("### 🚀 Trailing Settings")
    enable_trailing = st.sidebar.checkbox("Enable Trailing", value=True)
    
    if enable_trailing:
        trailing_activation_pct = st.sidebar.slider(
            "Activation %", 
            min_value=50, 
            max_value=95, 
            value=90, 
            step=5,
            help="Trailing starts when this % of target is reached"
        )
        trailing_target_increase = st.sidebar.slider(
            "Target Increase %", 
            min_value=1.0, 
            max_value=10.0, 
            value=2.0, 
            step=0.5,
            help="New target = Current Price × (1 + this %)"
        )
        trailing_sl_gap_pct = st.sidebar.slider(
            "SL Gap %", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.5, 
            step=0.1,
            help="Gap between trailing target and SL"
        )
    else:
        trailing_activation_pct = 90
        trailing_target_increase = 2.0
        trailing_sl_gap_pct = 0.5
    
    st.sidebar.markdown("---")
    
    # === POSITION LIMITS ===
    st.sidebar.markdown("### 📊 Position Limits")
    max_concurrent = st.sidebar.slider(
        "Max Concurrent Trades", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.max_concurrent_trades
    )
    
    if max_concurrent != st.session_state.max_concurrent_trades:
        st.session_state.max_concurrent_trades = max_concurrent
    
    positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
    open_count = len([p for p in positions if p['status'] == 'OPEN'])
    
    st.sidebar.metric("Open Positions", f"{open_count} / {max_concurrent}")
    
    if open_count >= max_concurrent:
        st.sidebar.warning("⚠️ Max trades reached")
    
    st.sidebar.markdown("---")
    
    # === SCANNER ===
    st.sidebar.markdown("### 🔍 Stock Scanner")
    
    if not st.session_state.option_stocks:
        if st.sidebar.button("🔎 Scan F&O Stocks", use_container_width=True):
            with st.spinner("Fetching F&O stocks..."):
                stocks = get_all_option_stocks(st.session_state.kite)
                st.session_state.option_stocks = stocks
                st.success(f"Found {len(stocks)} F&O stocks!")
                st.rerun()
    else:
        st.sidebar.success(f"✅ {len(st.session_state.option_stocks)} stocks loaded")
        
        if st.sidebar.button("🔄 Refresh Stock List", use_container_width=True):
            st.session_state.option_stocks = []
            st.rerun()
    
    # Trading Controls
    if st.session_state.option_stocks:
        if st.session_state.daily_target_reached:
            st.sidebar.success("🎉 Daily target reached! Trading stopped.")
        elif st.session_state.daily_sl_reached:
            st.sidebar.error("🛑 Daily stop loss hit! Trading stopped.")
        elif not st.session_state.trading_active:
            if st.sidebar.button("▶️ START TRADING", type="primary", use_container_width=True):
                st.session_state.trading_active = True
                st.rerun()
        else:
            if st.sidebar.button("⏹️ STOP TRADING", use_container_width=True):
                st.session_state.trading_active = False
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # === SIGNAL QUALITY INFO ===
    st.sidebar.markdown("### 📊 Signal Quality Requirements")
    st.sidebar.caption("""
    ✅ Min Score: 15 (was 12)
    ✅ Min Confirmations: 6 (was 4)
    ✅ Min Categories: 4 different
    ✅ Min R:R: 1.5:1
    ✅ Expiry Day: Avoided
    """)
    
    st.sidebar.markdown("---")
    
    # === RESET ===
    if st.sidebar.button("🔄 Reset All", use_container_width=True):
        st.session_state.paper_positions = []
        st.session_state.live_positions = []
        st.session_state.paper_capital = st.session_state.initial_paper_capital
        st.session_state.trade_logs = []
        st.session_state.active_signals = []
        st.session_state.daily_target_reached = False
        st.session_state.daily_sl_reached = False
        st.session_state.trading_stopped_reason = None
        st.session_state.daily_realized_pnl = 0
        st.session_state.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
        st.rerun()
    
    # Return all settings
    return {
        'target_pct': target_pct,
        'stop_loss_pct': stop_loss_pct,
        'max_capital_risk': max_capital_risk,
        'enable_trailing': enable_trailing,
        'trailing_activation_pct': trailing_activation_pct,
        'trailing_target_increase': trailing_target_increase,
        'trailing_sl_gap_pct': trailing_sl_gap_pct,
        'avoid_expiry_day': avoid_expiry_day,
        'rr_ratio': rr_ratio,
        'index_only_mode': index_only_mode,
        'close_all_on_daily_sl': close_all_on_daily_sl,
        'daily_profit_target': daily_profit_target,
        'daily_stop_loss': daily_stop_loss
    }

# ============================================================================
# END OF CHUNK 8 (UPDATED FOR OPTION 1)
# Changes Summary:
# 1. Completely redesigned render_login_page() with 2-step process
# 2. Step 1: API Key and Secret input form with validation
# 3. Step 2: Generate login URL and token exchange
# 4. Added session info display in sidebar
# 5. Added Logout button in sidebar
# 6. WebSocket connection now uses st.session_state.api_key
# 7. Added credential validation before WebSocket connect
# 8. Added "Clear All & Start Fresh" button
# ============================================================================



# ============================================================================
# CHUNK 9: MAIN APPLICATION UI - TABS (UPDATED - GTT Monitoring)
# Description: Tab-based UI for signals, positions, logs, charts, and GTT Monitor
# Changes Made:
#   - ADDED: render_gtt_monitor_tab() - New tab to monitor GTT orders
#   - Shows all active GTT orders
#   - Shows GTT status for each position
#   - Allows manual GTT cancellation
#   - Shows GTT history/triggered orders
# Dependencies: Chunk 1-8
# ============================================================================

def render_signals_tab():
    """Render the Signals tab with real-time option prices."""
    st.subheader("🔍 Active Signals (Real-Time Prices)")
    
    if st.session_state.active_signals:
        positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
        open_symbols = [p['symbol'] for p in positions if p['status'] == 'OPEN']
        
        st.info(f"📊 Found {len(st.session_state.active_signals)} signals | Open: {len(open_symbols)} / Max: {st.session_state.max_concurrent_trades}")
        
        signals_data = []
        for idx, sig in enumerate(st.session_state.active_signals[:20]):
            lot_size = sig.get('lot_size', 1)
            premium = sig.get('option_premium', 0)
            cost_per_lot = premium * lot_size
            
            signals_data.append({
                'Rank': idx + 1,
                'Status': '✅ TAKEN' if sig['symbol'] in open_symbols else '⏳ WAITING',
                'Symbol': sig['symbol'],
                'Signal': sig['signal'],
                'Spot ₹': f"₹{sig['spot_price']:.2f}",
                'Strike': sig.get('strike', 'N/A'),
                'Premium ₹': f"₹{premium:.2f}",
                'Lot Size': lot_size,
                'Cost/Lot': f"₹{cost_per_lot:,.2f}",
                'Confidence': f"{sig['confidence']:.0f}%",
                'Strength': sig['strength'],
                'Source': sig.get('price_source', 'N/A'),
                'Contract': sig.get('option_contract', 'N/A')[:20]
            })
        
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, use_container_width=True, height=400)
        
        # Show calculation example
        if st.session_state.active_signals:
            st.markdown("---")
            st.markdown("### 💡 Trade Calculation (Top Signal)")
            sig = st.session_state.active_signals[0]
            premium = sig.get('option_premium', 0)
            lot_size = sig.get('lot_size', 1)
            cost_per_lot = premium * lot_size
            
            st.info(f"""
            **{sig['symbol']} {sig['signal']}**
            - Contract: {sig.get('option_contract', 'N/A')}
            - Strike: ₹{sig.get('strike', 'N/A')}
            - Premium: ₹{premium:.2f} (Source: {sig.get('price_source', 'N/A')})
            - Lot Size: {lot_size}
            - Investment per lot: ₹{premium:.2f} × {lot_size} = **₹{cost_per_lot:,.2f}**
            """)
    else:
        st.info("No signals yet. Start trading to see signals with real-time option prices.")


def render_positions_tab():
    """Render the Positions overview tab."""
    st.subheader("💼 All Positions Overview")
    
    positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
    open_positions = [p for p in positions if p['status'] == 'OPEN']
    closed_positions = [p for p in positions if p['status'] == 'CLOSED']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Open", len(open_positions))
    with col2:
        st.metric("Closed", len(closed_positions))
    with col3:
        st.metric("Total", len(positions))
    
    if open_positions:
        st.markdown("### 🟢 Open Positions")
        open_data = []
        for p in open_positions:
            open_data.append({
                'ID': p['id'],
                'Symbol': p['symbol'],
                'Type': p['signal'],
                'Contract': p.get('contract', 'N/A')[:25],
                'Entry ₹': f"₹{p['entry_price']:.2f}",
                'Current ₹': f"₹{p['current_price']:.2f}",
                'Target ₹': f"₹{p.get('target_price', 0):.2f}",
                'SL ₹': f"₹{p.get('sl_price', 0):.2f}",
                'Invested': f"₹{p['investment']:,.2f}",
                'P&L': f"₹{p['pnl']:.2f}",
                'P&L %': f"{p['pnl_pct']:.2f}%",
                'GTT': p.get('gtt_status', 'N/A') if not st.session_state.paper_mode else 'Paper',
                'Source': p.get('price_source', 'N/A')
            })
        st.dataframe(pd.DataFrame(open_data), use_container_width=True)
    
    if closed_positions:
        st.markdown("### ⚫ Recent Closed Positions")
        closed_data = []
        for p in closed_positions[-20:]:
            closed_data.append({
                'ID': p['id'],
                'Symbol': p['symbol'],
                'Type': p['signal'],
                'Entry ₹': f"₹{p['entry_price']:.2f}",
                'Exit ₹': f"₹{p.get('exit_price', 0):.2f}",
                'P&L': f"₹{p['pnl']:.2f}",
                'P&L %': f"{p['pnl_pct']:.2f}%",
                'Reason': p.get('exit_reason', 'N/A')
            })
        st.dataframe(pd.DataFrame(closed_data), use_container_width=True)


def render_live_positions_tab(target_pct, stop_loss_pct):
    """Render detailed live/paper positions tab."""
    mode = "Paper" if st.session_state.paper_mode else "Live"
    st.subheader(f"{'📄' if st.session_state.paper_mode else '🔴'} {mode} Positions - Detailed View")
    
    positions = st.session_state.paper_positions if st.session_state.paper_mode else st.session_state.live_positions
    open_positions = [p for p in positions if p['status'] == 'OPEN']
    
    if open_positions:
        st.success(f"✅ {len(open_positions)} position(s) currently open")
        
        for p in open_positions:
            pnl_emoji = "🟢" if p['pnl'] >= 0 else "🔴"
            
            # GTT status indicator
            gtt_status = p.get('gtt_status', 'N/A')
            if gtt_status == 'active':
                gtt_indicator = "🟢 GTT Active"
            elif gtt_status == 'triggered':
                gtt_indicator = "✅ GTT Triggered"
            elif gtt_status == 'failed':
                gtt_indicator = "❌ GTT Failed"
            else:
                gtt_indicator = f"⚪ GTT: {gtt_status}"
            
            with st.expander(f"{pnl_emoji} {p['symbol']} {p['signal']} | P&L: ₹{p['pnl']:.2f} ({p['pnl_pct']:.2f}%) | {gtt_indicator}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Entry Details**")
                    st.write(f"Symbol: {p['symbol']}")
                    st.write(f"Signal: {p['signal']}")
                    st.write(f"Contract: {p.get('contract', 'N/A')}")
                    st.write(f"Strike: ₹{p.get('strike', 'N/A')}")
                    st.write(f"Time: {p['timestamp'].strftime('%H:%M:%S')}")
                
                with col2:
                    st.markdown("**Pricing**")
                    st.write(f"Entry: ₹{p['entry_price']:.2f}")
                    st.write(f"Current: ₹{p['current_price']:.2f}")
                    st.write(f"Spot: ₹{p.get('spot_price', 0):.2f}")
                    st.write(f"Source: {p.get('price_source', 'N/A')}")
                
                with col3:
                    st.markdown("**GTT OCO Levels**")
                    st.write(f"🎯 Target: ₹{p.get('target_price', 0):.2f}")
                    st.write(f"🛑 Stop Loss: ₹{p.get('sl_price', 0):.2f}")
                    st.write(f"GTT Status: {gtt_status}")
                    if p.get('gtt_oco_id'):
                        st.write(f"GTT ID: {p['gtt_oco_id']}")
                
                with col4:
                    st.markdown("**P&L**")
                    st.write(f"Invested: ₹{p['investment']:,.2f}")
                    st.write(f"Current: ₹{p['current_value']:,.2f}")
                    st.write(f"P&L: {pnl_emoji} ₹{p['pnl']:.2f}")
                    st.write(f"P&L %: {p['pnl_pct']:.2f}%")
                    st.write(f"Max Profit: ₹{p['max_profit']:.2f}")
                
                # Progress bar
                target_value = p['investment'] * (1 + target_pct / 100)
                sl_value = p['investment'] * (1 - stop_loss_pct / 100)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🛑 Stop Loss", f"₹{p.get('sl_price', sl_value):,.2f}")
                with col2:
                    st.metric("Current Value", f"₹{p['current_value']:,.2f}")
                with col3:
                    st.metric("🎯 Target", f"₹{p.get('target_price', target_value):,.2f}")
                
                progress = min(max(p['pnl_pct'] / target_pct, -1), 1)
                if progress >= 0:
                    st.progress(progress)
                    st.caption(f"Progress to target: {progress * 100:.1f}%")
                else:
                    st.progress(0)
                    st.caption(f"⚠️ Loss: {p['pnl_pct']:.1f}%")
        
        # Summary
        st.markdown("---")
        st.markdown("### 📊 Summary")
        
        total_invested = sum(p['investment'] for p in open_positions)
        total_value = sum(p['current_value'] for p in open_positions)
        total_pnl = sum(p['pnl'] for p in open_positions)
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        with col2:
            st.metric("Current Value", f"₹{total_value:,.2f}")
        with col3:
            st.metric("Unrealized P&L", f"₹{total_pnl:,.2f}")
        with col4:
            st.metric("Total P&L %", f"{total_pnl_pct:.2f}%")
    else:
        st.info("No open positions. Start trading to see positions here.")


def render_gtt_monitor_tab():
    """
    NEW: Render GTT Monitoring tab.
    
    Features:
    - Display all active GTT orders from Zerodha
    - Show GTT status for each open position
    - Allow manual GTT cancellation
    - Sync GTT status with positions
    """
    st.subheader("📋 GTT Order Monitor")
    
    # Check if in paper mode
    if st.session_state.paper_mode:
        st.warning("⚠️ GTT monitoring is only available in LIVE trading mode. Paper trading doesn't use GTT orders.")
        st.info("Switch to Live mode from sidebar to see GTT orders.")
        return
    
    # Check if logged in
    if not st.session_state.kite or not st.session_state.access_token:
        st.error("❌ Not connected to Zerodha. Please login first.")
        return
    
    kite = st.session_state.kite
    
    # === SECTION 1: SYNC & REFRESH ===
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh GTT Orders", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🔗 Sync GTT Status", use_container_width=True):
            with st.spinner("Syncing GTT status..."):
                summary = sync_gtt_status_all_positions(kite)
                st.success(f"Synced! Active: {summary['active']} | Triggered: {summary['triggered']} | Unknown: {summary['unknown']}")
    
    st.markdown("---")
    
    # === SECTION 2: GTT STATUS BY POSITION ===
    st.markdown("### 🎯 GTT Status by Position")
    
    live_positions = st.session_state.live_positions
    open_positions = [p for p in live_positions if p['status'] == 'OPEN']
    
    if open_positions:
        gtt_position_data = []
        
        for p in open_positions:
            gtt_id = p.get('gtt_oco_id', p.get('gtt_target_id', 'N/A'))
            gtt_status = p.get('gtt_status', 'unknown')
            
            # Status emoji
            if gtt_status == 'active':
                status_display = "🟢 Active"
            elif gtt_status == 'triggered':
                status_display = "✅ Triggered"
            elif gtt_status == 'cancelled':
                status_display = "🗑️ Cancelled"
            elif gtt_status == 'failed':
                status_display = "❌ Failed"
            else:
                status_display = f"⚪ {gtt_status}"
            
            gtt_position_data.append({
                'Position ID': p['id'],
                'Symbol': p['symbol'],
                'Signal': p['signal'],
                'Entry ₹': f"₹{p['entry_price']:.2f}",
                'Target ₹': f"₹{p.get('target_price', 0):.2f}",
                'SL ₹': f"₹{p.get('sl_price', 0):.2f}",
                'GTT ID': gtt_id if gtt_id else 'None',
                'GTT Type': p.get('gtt_type', 'OCO'),
                'GTT Status': status_display,
                'Current P&L': f"₹{p['pnl']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(gtt_position_data), use_container_width=True)
        
        # Manual GTT cancellation for positions
        st.markdown("#### 🗑️ Cancel GTT for Position")
        
        position_options = {f"{p['id']} - {p['symbol']} ({p['signal']})": p for p in open_positions if p.get('gtt_oco_id')}
        
        if position_options:
            selected_position = st.selectbox(
                "Select Position",
                options=list(position_options.keys()),
                key="cancel_position_select"
            )
            
            if st.button("🗑️ Cancel Selected GTT", type="secondary"):
                pos = position_options[selected_position]
                gtt_id = pos.get('gtt_oco_id')
                
                if gtt_id:
                    result = cancel_gtt_order(kite, gtt_id)
                    if result['success']:
                        pos['gtt_status'] = 'cancelled'
                        pos['gtt_oco_id'] = None
                        st.success(f"✅ GTT {gtt_id} cancelled for {pos['symbol']}")
                        log = f"{datetime.now().strftime('%H:%M:%S')} | 🗑️ GTT CANCELLED (Manual) | {pos['symbol']} | ID: {gtt_id}"
                        st.session_state.trade_logs.append(log)
                    else:
                        st.error(f"❌ {result['message']}")
                else:
                    st.warning("No GTT found for this position")
        else:
            st.info("No positions with active GTT orders to cancel")
    else:
        st.info("No open positions with GTT orders")
    
    st.markdown("---")
    
    # === SECTION 3: ALL GTT ORDERS FROM BROKER ===
    st.markdown("### 📊 All GTT Orders from Zerodha")
    
    try:
        with st.spinner("Fetching GTT orders..."):
            all_gtts = get_gtt_orders(kite)
        
        if all_gtts:
            # Summary metrics
            active_gtts = [g for g in all_gtts if g['status'] == 'active']
            triggered_gtts = [g for g in all_gtts if g['status'] in ['triggered', 'complete']]
            cancelled_gtts = [g for g in all_gtts if g['status'] in ['cancelled', 'deleted']]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total GTTs", len(all_gtts))
            with col2:
                st.metric("🟢 Active", len(active_gtts))
            with col3:
                st.metric("✅ Triggered", len(triggered_gtts))
            with col4:
                st.metric("🗑️ Cancelled", len(cancelled_gtts))
            
            # Filter options
            st.markdown("#### Filter GTT Orders")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=['active', 'triggered', 'cancelled', 'deleted', 'complete'],
                    default=['active']
                )
            
            with filter_col2:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=['OCO', 'SINGLE'],
                    default=['OCO', 'SINGLE']
                )
            
            # Filter GTTs
            filtered_gtts = [
                g for g in all_gtts
                if g['status'] in status_filter and g['gtt_type'] in type_filter
            ]
            
            if filtered_gtts:
                gtt_data = []
                
                for gtt in filtered_gtts:
                    # Format triggers
                    if gtt['gtt_type'] == 'OCO':
                        triggers = f"SL: ₹{gtt.get('sl_trigger', 0):.2f} | Target: ₹{gtt.get('target_trigger', 0):.2f}"
                    else:
                        triggers = f"₹{gtt.get('trigger_price', 0):.2f}"
                    
                    # Status emoji
                    status = gtt['status']
                    if status == 'active':
                        status_display = "🟢 Active"
                    elif status in ['triggered', 'complete']:
                        status_display = "✅ Triggered"
                    elif status in ['cancelled', 'deleted']:
                        status_display = "🗑️ Cancelled"
                    else:
                        status_display = f"⚪ {status}"
                    
                    gtt_data.append({
                        'GTT ID': gtt['id'],
                        'Symbol': gtt['tradingsymbol'][:20] if gtt['tradingsymbol'] else 'N/A',
                        'Type': gtt['gtt_type'],
                        'Triggers': triggers,
                        'Quantity': gtt.get('quantity', 0),
                        'Status': status_display,
                        'Created': str(gtt.get('created_at', 'N/A'))[:19],
                        'Last Price': f"₹{gtt.get('last_price', 0):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(gtt_data), use_container_width=True, height=400)
                
                # Bulk cancel option
                st.markdown("#### 🗑️ Cancel GTT Order")
                
                gtt_options = {f"{g['id']} - {g['tradingsymbol'][:15]}": g['id'] for g in filtered_gtts if g['status'] == 'active'}
                
                if gtt_options:
                    selected_gtt = st.selectbox(
                        "Select GTT to Cancel",
                        options=list(gtt_options.keys()),
                        key="cancel_gtt_select"
                    )
                    
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if st.button("🗑️ Cancel GTT", type="secondary"):
                            gtt_id = gtt_options[selected_gtt]
                            result = cancel_gtt_order(kite, gtt_id)
                            
                            if result['success']:
                                st.success(f"✅ {result['message']}")
                                log = f"{datetime.now().strftime('%H:%M:%S')} | 🗑️ GTT CANCELLED (Manual) | ID: {gtt_id}"
                                st.session_state.trade_logs.append(log)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
                else:
                    st.info("No active GTT orders to cancel")
            else:
                st.info("No GTT orders matching the filter criteria")
        else:
            st.info("No GTT orders found in your Zerodha account")
            
    except Exception as e:
        st.error(f"❌ Error fetching GTT orders: {str(e)}")
        st.caption("Make sure you're logged in and have proper API permissions")
    
    st.markdown("---")
    
    # === SECTION 4: GTT INFO ===
    with st.expander("ℹ️ About GTT OCO Orders"):
        st.markdown("""
        ### What is GTT OCO?
        
        **GTT (Good Till Triggered)** orders are conditional orders that remain active until triggered or cancelled.
        
        **OCO (One Cancels Other)** means:
        - Two orders are placed together (Target + Stop Loss)
        - When one triggers, the other is automatically cancelled
        - Only one exit order will execute
        
        ### GTT Status Meanings:
        
        | Status | Meaning |
        |--------|---------|
        | 🟢 Active | GTT is live and monitoring price |
        | ✅ Triggered | Price hit trigger, order executed |
        | 🗑️ Cancelled | GTT was manually cancelled |
        | ❌ Failed | GTT placement failed |
        
        ### Important Notes:
        
        1. **GTT Limits**: Zerodha allows max 20 active GTT orders per account
        2. **MIS + GTT**: GTT may not work if MIS position squares off at 3:20 PM
        3. **Price Buffer**: Keep some buffer between trigger and limit price
        4. **Validity**: GTT orders are valid for 1 year from placement
        """)


def render_logs_tab():
    """Render the Trade Logs tab."""
    st.subheader("📜 Trade Logs")
    
    if st.session_state.trade_logs:
        # Filter options
        col1, col2 = st.columns([1, 3])
        with col1:
            log_filter = st.selectbox(
                "Filter Logs",
                options=['All', 'Trades Only', 'GTT Only', 'Errors Only'],
                key="log_filter"
            )
        
        # Filter logs based on selection
        logs = st.session_state.trade_logs
        
        if log_filter == 'Trades Only':
            logs = [l for l in logs if 'TRADE' in l or 'ENTRY' in l or 'CLOSED' in l]
        elif log_filter == 'GTT Only':
            logs = [l for l in logs if 'GTT' in l]
        elif log_filter == 'Errors Only':
            logs = [l for l in logs if 'ERROR' in l or 'FAILED' in l or '❌' in l]
        
        log_text = "\n".join(reversed(logs[-100:]))
        st.text_area("Recent Activity", value=log_text, height=500)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Logs"):
                st.download_button(
                    "Download Full Logs",
                    "\n".join(st.session_state.trade_logs),
                    "trade_logs.txt",
                    "text/plain"
                )
        with col2:
            if st.button("🗑️ Clear Logs"):
                st.session_state.trade_logs = []
                st.rerun()
    else:
        st.info("No trade logs yet.")


def render_charts_tab():
    """Render the Charts tab."""
    st.subheader("📈 Charts")
    
    if st.session_state.option_stocks:
        selected = st.selectbox("Select Symbol", [s['symbol'] for s in st.session_state.option_stocks])
        
        col1, col2 = st.columns(2)
        with col1:
            interval = st.selectbox("Interval", ['5minute', '15minute', '30minute', 'day'])
        with col2:
            days = st.number_input("Days", min_value=1, max_value=30, value=5)
        
        if st.button("📊 Load Chart"):
            stock = next((s for s in st.session_state.option_stocks if s['symbol'] == selected), None)
            
            if stock:
                with st.spinner(f"Loading {selected}..."):
                    analyzer = ProfessionalTradingAnalysis(
                        st.session_state.kite,
                        selected,
                        stock['token'],
                        st.session_state.ws_manager
                    )
                    
                    df = analyzer.get_historical_data(days=days, interval=interval)
                    
                    if df is not None and len(df) > 0:
                        # Candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=df['date'],
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close']
                        )])
                        
                        fig.update_layout(
                            title=f"{selected} - {interval}",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current real-time price
                        ws_manager = st.session_state.ws_manager
                        if ws_manager and ws_manager.is_connected:
                            ltp = ws_manager.get_ltp(stock['token'])
                            if ltp > 0:
                                st.metric("Real-Time Price", f"₹{ltp:.2f}")
                    else:
                        st.error("Could not load chart data")
    else:
        st.info("Scan stocks first to load charts")


def render_websocket_tab():
    """Render WebSocket monitoring tab."""
    st.subheader("📡 WebSocket Monitor")
    
    ws_manager = st.session_state.ws_manager
    
    if ws_manager and ws_manager.is_connected:
        st.success(f"🟢 WebSocket Connected | Subscribed: {ws_manager.get_subscribed_count()} tokens")
        
        # Show live ticks
        all_ticks = ws_manager.get_all_ticks()
        
        if all_ticks:
            st.markdown("### 📊 Live Tick Data")
            
            tick_data = []
            for token, tick in all_ticks.items():
                tick_data.append({
                    'Token': token,
                    'LTP': f"₹{tick.get('ltp', 0):.2f}",
                    'Change %': f"{tick.get('change', 0):.2f}%",
                    'Volume': tick.get('volume', 0),
                    'Bid': f"₹{tick.get('bid', 0):.2f}",
                    'Ask': f"₹{tick.get('ask', 0):.2f}",
                    'Updated': tick.get('timestamp', datetime.now()).strftime('%H:%M:%S')
                })
            
            st.dataframe(pd.DataFrame(tick_data), use_container_width=True)
        else:
            st.info("No ticks received yet. Subscribe to tokens to see live data.")
        
        # Manual subscription
        st.markdown("---")
        st.markdown("### 🔧 Manual Token Subscription")
        
        tokens_input = st.text_input("Enter tokens (comma-separated)", placeholder="256265, 260105")
        
        if st.button("Subscribe Tokens"):
            if tokens_input:
                try:
                    tokens = [int(t.strip()) for t in tokens_input.split(",")]
                    ws_manager.subscribe_tokens(tokens, mode='full')
                    st.success(f"Subscribed to {len(tokens)} tokens")
                except:
                    st.error("Invalid token format")
    else:
        st.warning("WebSocket not connected. Connect from sidebar.")

# ============================================================================
# END OF CHUNK 9 (UPDATED)
# Changes Summary:
# 1. Added render_gtt_monitor_tab() - Complete GTT monitoring UI
# 2. Shows GTT status for each open position
# 3. Displays all GTT orders from Zerodha
# 4. Allows manual GTT cancellation
# 5. Filter GTTs by status and type
# 6. Added GTT info/help section
# 7. Updated render_positions_tab() to show GTT status
# 8. Updated render_live_positions_tab() to show GTT details
# 9. Added log filtering in render_logs_tab()
# ============================================================================

# ============================================================================
# CHUNK 10: MAIN FUNCTION AND APP ENTRY POINT (UPDATED - GTT Monitor Tab)
# Description: Main application function that ties everything together
# Changes Made:
#   - ADDED: GTT Monitor tab in the tabs list
#   - Updated tab count from 6 to 7
# Dependencies: Chunk 1-9
# ============================================================================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Professional Options Trading Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title
    st.title("🤖 Professional Options Trading Bot")
    st.markdown("*Real-Time Options Scanner & Trading System with WebSocket + GTT OCO*")
    
    # === DAILY TARGET/SL BANNERS ===
    if st.session_state.daily_target_reached:
        st.balloons()
        st.success("""
        🎉🎉🎉 **CONGRATULATIONS! DAILY PROFIT TARGET REACHED!** 🎉🎉🎉
        
        You've achieved your daily profit goal. Trading has been automatically stopped.
        
        Take a break, enjoy your success, and come back tomorrow! 💰
        """)
        
        pnl_info = calculate_daily_pnl()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Today's Profit", f"₹{pnl_info['total_pnl']:,.2f}", delta="TARGET HIT! ✅")
        with col2:
            st.metric("Target Was", f"₹{st.session_state.daily_profit_target:,.2f}")
        with col3:
            st.metric("Exceeded By", f"₹{pnl_info['total_pnl'] - st.session_state.daily_profit_target:,.2f}")
    
    elif st.session_state.daily_sl_reached:
        st.error("""
        🛑 **DAILY STOP LOSS HIT - TRADING STOPPED** 🛑
        
        Your daily loss limit has been reached. Trading has been automatically stopped to protect your capital.
        
        Take a break, review your trades, and come back tomorrow with a fresh mindset. 🧘
        """)
        
        pnl_info = calculate_daily_pnl()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Today's Loss", f"₹{pnl_info['total_pnl']:,.2f}", delta="LIMIT HIT! 🛑")
        with col2:
            st.metric("Max Loss Was", f"-₹{st.session_state.daily_stop_loss:,.2f}")
        with col3:
            st.metric("Trades Today", st.session_state.performance_stats['total_trades'])
    
    # Market status
    current_time = datetime.now().time()
    expiry_info = check_global_expiry_day()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if current_time < MARKET_OPEN.time():
            st.info(f"⏰ Market opens at {MARKET_OPEN.time().strftime('%H:%M')}")
        elif current_time > MARKET_CLOSE.time():
            st.error("🔴 Market Closed")
        else:
            st.success("🟢 Market Open")
    
    with col2:
        st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))
    
    with col3:
        if st.session_state.ws_manager and st.session_state.ws_manager.is_connected:
            st.success("📡 WebSocket: Connected")
        else:
            st.warning("📡 WebSocket: Disconnected")
    
    with col4:
        if expiry_info['is_expiry_day']:
            st.warning(f"📅 Expiry: {', '.join(expiry_info['expiring_indices'])}")
        else:
            st.info("📅 No Expiry Today")
    
    # Daily P&L status in header
    with col5:
        pnl_info = calculate_daily_pnl()
        total_pnl = pnl_info['total_pnl']
        if total_pnl >= 0:
            st.success(f"💰 P&L: ₹{total_pnl:,.0f}")
        else:
            st.error(f"📉 P&L: ₹{total_pnl:,.0f}")
    
    # Login check
    if st.session_state.access_token is None:
        render_login_page()
        return
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main content area
    if st.session_state.daily_target_reached:
        st.info("🎉 Trading completed for today - Daily target reached!")
    elif st.session_state.daily_sl_reached:
        st.warning("🛑 Trading stopped for today - Daily stop loss hit")
    elif st.session_state.trading_active:
        # Show trading status with expiry info
        if settings.get('avoid_expiry_day', True) and expiry_info['is_expiry_day']:
            st.success(f"🟢 TRADING ACTIVE - Scanning (Avoiding {', '.join(expiry_info['expiring_indices'])} expiry)")
        else:
            st.success("🟢 TRADING ACTIVE - Scanning for opportunities...")
        
        # Show daily limits status
        profit_target = settings.get('daily_profit_target', 0)
        daily_sl = settings.get('daily_stop_loss', 0)
        
        if profit_target > 0 or daily_sl > 0:
            limit_col1, limit_col2, limit_col3 = st.columns(3)
            with limit_col1:
                st.metric("Today's P&L", f"₹{pnl_info['total_pnl']:,.2f}")
            with limit_col2:
                if profit_target > 0:
                    remaining = profit_target - pnl_info['total_pnl']
                    st.metric("To Target", f"₹{remaining:,.2f}" if remaining > 0 else "✅ REACHED")
            with limit_col3:
                if daily_sl > 0:
                    buffer = daily_sl + pnl_info['total_pnl']
                    st.metric("SL Buffer", f"₹{buffer:,.2f}" if buffer > 0 else "🛑 HIT")
        
        # Run scan and trade with all parameters
        scan_and_trade(
            kite=st.session_state.kite,
            option_stocks=st.session_state.option_stocks,
            target_pct=settings['target_pct'],
            stop_loss_pct=settings['stop_loss_pct'],
            max_capital_risk_pct=settings['max_capital_risk'],
            max_concurrent_trades=st.session_state.max_concurrent_trades,
            enable_trailing=settings['enable_trailing'],
            trailing_activation_pct=settings['trailing_activation_pct'],
            trailing_target_increase=settings['trailing_target_increase'],
            trailing_sl_gap_pct=settings['trailing_sl_gap_pct'],
            ws_manager=st.session_state.ws_manager,
            avoid_expiry_day=settings.get('avoid_expiry_day', True),
            close_all_on_daily_sl=settings.get('close_all_on_daily_sl', False)
        )
    else:
        st.info("⏸️ TRADING PAUSED - Click 'Start Trading' to begin")
    
    # === TABS (UPDATED - Added GTT Monitor) ===
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Signals",
        "💼 Positions",
        "🔴 Live View" if not st.session_state.paper_mode else "📄 Paper View",
        "📋 GTT Monitor",  # NEW TAB
        "📜 Logs",
        "📈 Charts",
        "📡 WebSocket"
    ])
    
    with tab1:
        render_signals_tab()
    
    with tab2:
        render_positions_tab()
    
    with tab3:
        render_live_positions_tab(settings['target_pct'], settings['stop_loss_pct'])
    
    with tab4:
        render_gtt_monitor_tab()  # NEW TAB CONTENT
    
    with tab5:
        render_logs_tab()
    
    with tab6:
        render_charts_tab()
    
    with tab7:
        render_websocket_tab()
    
    # Auto-refresh when trading active (but not when limits hit)
    if st.session_state.trading_active and not st.session_state.kill_switch:
        if not st.session_state.daily_target_reached and not st.session_state.daily_sl_reached:
            time.sleep(3)
            st.rerun()


# ============================================================================
# APP ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

# ============================================================================
# END OF CHUNK 10 (UPDATED)
# Changes Summary:
# 1. Added GTT Monitor tab (tab4)
# 2. Updated tabs count from 6 to 7
# 3. Updated subtitle to mention GTT OCO
# ============================================================================
