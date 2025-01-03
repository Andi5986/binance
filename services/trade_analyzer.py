import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from typing import Union, Tuple, Callable
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.serialization
import os

from utils.data_processor import DataProcessor
from models.price_predictor import PricePredictor

@dataclass
class Condition:
    """Data class to hold condition information"""
    value: float
    threshold: Union[float, Tuple[float, float]]  # Can be either float or tuple of floats
    compare_func: Callable  # or more specifically: Callable[[float, Union[float, Tuple[float, float]]], bool]
    met: bool = False

    def __post_init__(self):
        self.met = self.compare_func(self.value, self.threshold)

@dataclass
class TradeSignal:
    """Data class to hold trading signal information"""
    current_price: float
    entry_price: float
    target_price: float
    stop_price: float
    prediction: float
    market_conditions: Dict
    signal_found: bool
    risk_reward_ratio: float = 0.0
    potential_profit: float = 0.0
    maximum_risk: float = 0.0
    
    def __post_init__(self):
        """Calculate additional metrics after initialization"""
        if self.entry_price and self.target_price and self.stop_price:
            self.risk_reward_ratio = (self.target_price - self.entry_price) / (self.entry_price - self.stop_price)
            self.potential_profit = (self.target_price - self.entry_price) / self.entry_price * 100
            self.maximum_risk = (self.entry_price - self.stop_price) / self.entry_price * 100

class MarketAnalyzer:
    """Enhanced market analysis incorporating successful patterns"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Enhanced market condition analysis with successful patterns"""
        try:
            current = data.iloc[-1]
            
            # Volume analysis (comparing to moving average)
            volume_ma = data['volume'].rolling(20).mean().iloc[-1]
            volume_condition = Condition(
                current['volume'],
                volume_ma * 1.5,  # Require 50% above average volume
                lambda x, t: x > t
            )

            # RSI optimization (target neutral range 45-55)
            rsi_condition = Condition(
                current['rsi'],
                (45, 55),  # Optimal range from analysis
                lambda x, t: t[0] <= x <= t[1]
            )

            # Volatility management (ATR threshold)
            atr_condition = Condition(
                current['atr'],
                1.0,  # Successful trades had ATR < 1
                lambda x, t: x < t
            )

            # MACD signal improvement
            macd_condition = Condition(
                current['macd'],
                0,  # Looking for positive MACD with negative histogram
                lambda x, t: x > t and current['macd_hist'] < 0
            )

            # Bollinger Bands positioning
            bb_condition = Condition(
                current['close'],
                (current['bb_lower'], current['bb_middle']),
                lambda x, t: t[0] < x < t[1]  # Price between lower and middle band
            )

            # Trend strength (ADX)
            adx_condition = Condition(
                current['adx'],
                20,  # Moderate trend strength
                lambda x, t: x > t
            )

            # Moving average alignment
            ma_condition = Condition(
                current['ma7'],
                current['ma25'],
                lambda x, t: x > t  # Short-term MA above long-term MA
            )

            return {
                'volume_strength': volume_condition,
                'rsi_optimal': rsi_condition,
                'volatility_safe': atr_condition,
                'macd_signal': macd_condition,
                'bb_position': bb_condition,
                'trend_strength': adx_condition,
                'ma_alignment': ma_condition
            }
            
        except Exception as e:
            self.logger.error(f"Error in market condition analysis: {str(e)}")
            return {}

class TradeAnalyzer:
    """Enhanced service class for analyzing market conditions and generating trade signals"""
    
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.traders: Dict[str, DataProcessor] = {}
        self.models: Dict[str, PricePredictor] = {}
        self.market_analyzer = MarketAnalyzer(config)
        self._last_trading_signals = {}
        
    def initialize_traders(self) -> bool:
        """Initialize traders with security checks"""
        self.logger.info("Initializing traders...")
        initialization_failed = False
        
        for file_path in self.config.DATA_DIR.glob('*USDT_*.json'):
            try:
                symbol = file_path.stem.split('_')[0].upper()
                
                # Initialize data processor
                self.traders[symbol] = DataProcessor(file_path)
                
                # Load trained model with security checks
                model_path = self.config.MODEL_DIR / f'{symbol}_model.pth'
                if not model_path.exists():
                    self.logger.warning(f"No trained model found for {symbol}")
                    continue
                    
                model = self._load_model(model_path)
                if model:
                    self.models[symbol] = model
                    self.logger.info(f"✅ Successfully initialized {symbol} trader")
                else:
                    initialization_failed = True
                    
            except Exception as e:
                self.logger.error(f"❌ Error initializing {symbol}: {str(e)}")
                initialization_failed = True
                
        if not self.traders:
            self.logger.error("No traders were successfully initialized")
            return False
            
        self.logger.info(f"Total number of active traders: {len(self.traders)}")
        return not initialization_failed

    def _load_model(self, model_path: Path) -> Optional[PricePredictor]:
        """Load a trained model with enhanced security checks"""
        try:
            # Security checks
            model_path = Path(model_path).resolve()
            trusted_dir = Path(self.config.MODEL_DIR).resolve()
            
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
                
            if not str(model_path).startswith(str(trusted_dir)):
                raise ValueError(f"Model path {model_path} is outside trusted directory")
                
            if model_path.suffix.lower() != '.pth':
                raise ValueError(f"Invalid model file extension: {model_path.suffix}")
                
            # Size and permission checks
            max_size = 1024 * 1024 * 1024  # 1GB
            if os.path.getsize(model_path) > max_size:
                raise ValueError(f"Model file too large: {model_path}")
                
            file_mode = os.stat(model_path).st_mode
            if file_mode & 0o111:
                raise ValueError(f"Model file has executable permissions")
                
            # Load model
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            if not all(key in checkpoint for key in {'model_state', 'feature_cols'}):
                raise ValueError("Invalid checkpoint format")
                
            feature_count = len(checkpoint['feature_cols'])
            model = PricePredictor(feature_count)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def generate_trading_signals(self) -> Dict[str, TradeSignal]:
        """Generate trading signals with enhanced logging"""
        self.logger.info("Starting signal generation...")
        self._last_trading_signals = {}
        current_data = self.get_current_market_data()
        signals = {}
        total_analyzed = 0
        signals_found = 0
        
        for symbol, trader in self.traders.items():
            try:
                total_analyzed += 1
                signal = self._generate_single_signal(symbol, trader, current_data)
                
                if signal:
                    signals_found += 1
                    if self._validate_signal(signal):
                        signals[symbol] = signal
                        self.logger.info(f"✅ Valid signal generated for {symbol}")
                    else:
                        self.logger.debug(f"Signal validation failed for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
                continue
                
        # Log summary statistics
        self.logger.info(f"Signal generation complete:")
        self.logger.info(f"Total symbols analyzed: {total_analyzed}")
        self.logger.info(f"Initial signals found: {signals_found}")
        self.logger.info(f"Valid signals after filtering: {len(signals)}")
        
        self._last_trading_signals = signals
        return signals

    def _generate_single_signal(
        self, symbol: str, trader: DataProcessor, current_data: Dict
    ) -> Optional[TradeSignal]:
        """Generate enhanced trading signal with improved technical filters"""
        data = current_data.get(symbol)
        if data is None or data.empty:
            return None

        try:
            processed_data = trader.calculate_features(data)
            model = self.models.get(symbol)
            
            if model is None:
                return None

            # Get model prediction
            with torch.no_grad():
                features = processed_data[trader.feature_cols].iloc[-1:].values
                prediction = model(torch.FloatTensor(features)).item()

            # Enhanced market condition analysis
            market_conditions = self.market_analyzer.analyze_market_conditions(processed_data)
            
            # Check technical filters
            if not self._check_technical_filters(processed_data):
                return None

            current_price = self._get_current_price(symbol)
            if current_price is None:
                return None

            # Dynamic price targets based on volatility
            atr = processed_data['atr'].iloc[-1]
            entry_factor = self.config.ENTRY_PRICE_FACTOR
            target_factor = max(1.009, min(1.015, 1.009 + atr * 0.002))
            stop_factor = max(0.995, min(0.992, 0.995 - atr * 0.001))

            return TradeSignal(
                current_price=current_price,
                entry_price=current_price * entry_factor,
                target_price=current_price * entry_factor * target_factor,
                stop_price=current_price * entry_factor * stop_factor,
                prediction=prediction,
                market_conditions=market_conditions,
                signal_found=prediction > self.config.CONFIDENCE_THRESHOLD
            )

        except Exception as e:
            self.logger.error(f"Error in signal generation: {str(e)}")
            return None

    def _check_technical_filters(self, data: pd.DataFrame) -> bool:
        """Check technical filters with more lenient thresholds"""
        try:
            current = data.iloc[-1]
            
            # Volume filter - slightly reduced threshold
            volume_ma = data['volume'].rolling(20).mean().iloc[-1]
            if current['volume'] < volume_ma * 1.3:  # Reduced from 1.5
                return False

            # Wider RSI range
            if not (40 <= current['rsi'] <= 60):  # Widened from 45-55
                return False

            # Higher ATR threshold
            if current['atr'] > 1.5:  # Increased from 1.0
                return False

            # Modified MACD filter
            if not (current['macd'] > -0.1):  # Allow slightly negative MACD
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in technical filters: {str(e)}")
            return False

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate signal meets minimum criteria with more lenient thresholds"""
        try:
            # Log signal details for debugging
            self.logger.debug(f"Validating signal:")
            self.logger.debug(f"  Prediction: {signal.prediction}")
            self.logger.debug(f"  Risk/Reward: {signal.risk_reward_ratio}")
            self.logger.debug(f"  Market Conditions: {sum(1 for c in signal.market_conditions.values() if c.met)}/{len(signal.market_conditions)}")

            # Adjust minimum requirements
            # Only require volume and one more key condition to be met
            key_conditions = {
                'volume_strength',
                'volatility_safe',
                'rsi_optimal',
                'macd_signal'
            }
            conditions_met = sum(1 for name, cond in signal.market_conditions.items() 
                               if name in key_conditions and cond.met)
            
            # Volume must be met plus at least one other condition
            volume_met = signal.market_conditions.get('volume_strength', 
                        Condition(0, 0, lambda x, t: False)).met
            
            if not volume_met:
                self.logger.debug("Signal rejected: Volume condition not met")
                return False
                
            if conditions_met < 2:  # Volume plus at least one other condition
                self.logger.debug("Signal rejected: Insufficient conditions met")
                return False

            # Lower the prediction threshold slightly
            if signal.prediction < 0.75:  # Reduced from 0.8
                self.logger.debug("Signal rejected: Prediction too low")
                return False

            # More lenient risk/reward ratio
            if signal.risk_reward_ratio < 1.5:  # Reduced from 2.0
                self.logger.debug("Signal rejected: Risk/reward ratio too low")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in signal validation: {str(e)}")
            return False

    def get_current_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch and process current market data with enhanced logging"""
        self.logger.info("Fetching current market data for all symbols...")
        market_data = {}
        total_symbols = len(self.traders)
        processed = 0
        
        for symbol in self.traders:
            try:
                processed += 1
                self.logger.info(f"Processing {symbol} ({processed}/{total_symbols})")
                
                # Fetch klines
                self.logger.debug(f"{symbol}: Fetching klines...")
                klines = self._fetch_klines(symbol)
                if not klines:
                    self.logger.warning(f"{symbol}: No kline data available")
                    continue
                    
                # Fetch order book
                self.logger.debug(f"{symbol}: Fetching order book...")
                depth = self._fetch_order_book(symbol)
                if not depth:
                    self.logger.warning(f"{symbol}: No order book data available")
                    continue
                
                # Process data
                processed_data = self._process_market_data(klines, depth, symbol)
                if not processed_data.empty:
                    market_data[symbol] = processed_data
                else:
                    self.logger.warning(f"{symbol}: Empty processed data")
                
            except Exception as e:
                self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Market data processing complete. Processed {len(market_data)}/{total_symbols} symbols")
        return market_data

    def _fetch_klines(self, symbol: str) -> List[List]:
        """Fetch kline data with error handling"""
        try:
            return self.client.get_klines(
                symbol=symbol,
                interval=self.client.KLINE_INTERVAL_15MINUTE,
                limit=100
            )
        except Exception as e:
            self.logger.error(f"Error fetching klines for {symbol}: {str(e)}")
            return None

    def _fetch_order_book(self, symbol: str) -> Dict:
        """Fetch order book data with error handling"""
        try:
            return self.client.get_order_book(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with error handling"""
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def _process_market_data(self, klines: List[List], depth: Dict, symbol: str) -> pd.DataFrame:
        """Process market data with enhanced quality checks and missing value handling"""
        try:
            self.logger.info(f"Processing market data for {symbol}...")
            
            # Ensure klines has data
            if not klines or len(klines) == 0:
                self.logger.warning(f"{symbol}: No kline data available")
                return pd.DataFrame()
            
            # More flexible column naming and handling
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                'buy_quote_volume', 'ignore'
            ]
            
            # Create DataFrame with flexible column handling
            df = pd.DataFrame(klines, columns=columns[:len(klines[0])])
            
            # Initial data quality check
            if len(df) < 100:
                self.logger.warning(f"{symbol}: Insufficient data points: {len(df)}")
                return pd.DataFrame()
            
            # Convert numeric columns with validation
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        self.logger.warning(f"{symbol}: {col} has {null_count} null values")
                else:
                    self.logger.warning(f"{symbol}: Column {col} not found in data")
            
            # Ensure timestamp columns
            if 'open_time' not in df.columns:
                df['open_time'] = range(len(df))
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Validate price data
            df = df[df['close'] != 0]
            
            # Handle missing values with a more flexible approach
            df = self._handle_missing_values(df, symbol)
            if df.empty:
                return pd.DataFrame()
            
            # Add order book metrics with validation
            try:
                bid_depth = sum(float(bid[1]) for bid in depth['bids'][:10]) if depth and 'bids' in depth else 0
                ask_depth = sum(float(ask[1]) for ask in depth['asks'][:10]) if depth and 'asks' in depth else 0
                spread = ((float(depth['asks'][0][0]) - float(depth['bids'][0][0])) / float(depth['bids'][0][0])) if depth and 'asks' in depth and 'bids' in depth else 0
                
                df['bid_depth'] = bid_depth
                df['ask_depth'] = ask_depth
                df['total_liquidity'] = bid_depth + ask_depth
                df['spread_percentage'] = spread * 100
            except Exception as e:
                self.logger.error(f"{symbol}: Error processing order book: {str(e)}")
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df, symbol)
            if df.empty:
                return pd.DataFrame()
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Final data quality check
            final_quality_check = self._perform_final_quality_check(df, symbol)
            if not final_quality_check:
                return pd.DataFrame()
            
            df = df.sort_index()
            
            # Log final metrics
            self._log_final_metrics(df, symbol)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _handle_missing_values(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle missing values with proper validation"""
        try:
            # Identify key columns dynamically
            numeric_cols = ['close', 'high', 'low', 'volume']
            existing_cols = [col for col in numeric_cols if col in df.columns]
            
            # Add technical indicator columns if they exist
            technical_cols = ['ma7', 'ma25', 'rsi', 'macd', 'atr', 'adx']
            existing_cols.extend([col for col in technical_cols if col in df.columns])
            
            # Check for missing values in existing columns
            for col in existing_cols:
                missing = df[col].isnull().sum()
                if missing > 0:
                    self.logger.warning(f"{symbol}: Found {missing} missing values in {col}")
                    
                    if missing > len(df) * 0.1:  # More than 10% missing
                        self.logger.error(f"{symbol}: Too many missing values in {col}")
                        return pd.DataFrame()
            
            # Columns to fill
            fill_cols = [col for col in existing_cols if df[col].isnull().any()]
            
            # Forward and backward fill with limit
            for col in fill_cols:
                df[col] = df[col].ffill(limit=3).bfill(limit=3)
            
            # Check if any NaNs remain
            remaining_nulls = df[existing_cols].isnull().sum().sum()
            if remaining_nulls > 0:
                self.logger.warning(f"{symbol}: {remaining_nulls} null values remain after filling")
                
            return df
            
        except Exception as e:
            self.logger.error(f"{symbol}: Error handling missing values: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators with validation"""
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], period=14)
            rsi_last = df['rsi'].iloc[-1]
            self.logger.debug(f"{symbol}: RSI calculated. Last value: {rsi_last:.2f}")
            
            # MACD with validation
            macd_line, signal_line, macd_hist = self._calculate_macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = macd_hist
            self.logger.debug(f"{symbol}: MACD line range: {macd_line.min():.4f} to {macd_line.max():.4f}")
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (std * 2)
            df['bb_lower'] = df['bb_middle'] - (std * 2)
            
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            volume_ratio = df['volume'].iloc[-1] / df['volume_ma'].iloc[-1]
            self.logger.debug(f"{symbol}: Volume ratio: {volume_ratio:.2f}x average")
            
            # ATR with validation
            df['atr'] = self._calculate_atr(df, period=14)
            atr_pct = df['atr'].iloc[-1] / df['close'].iloc[-1] * 100
            self.logger.debug(f"{symbol}: ATR percentage: {atr_pct:.2f}%")
            
            # ADX with validation
            df['adx'] = self._calculate_adx(df, period=14)
            adx_last = df['adx'].iloc[-1]
            self.logger.debug(f"{symbol}: ADX trend strength: {adx_last:.2f}")
            
            # Add moving averages
            df['ma7'] = df['close'].rolling(window=7).mean()
            df['ma25'] = df['close'].rolling(window=25).mean()
            
            # Validate moving averages
            ma7_last = df['ma7'].iloc[-1]
            ma25_last = df['ma25'].iloc[-1]
            self.logger.debug(f"{symbol}: MA7: {ma7_last:.4f}, MA25: {ma25_last:.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"{symbol}: Error calculating indicators: {str(e)}")
            return pd.DataFrame()

    def _perform_final_quality_check(self, df: pd.DataFrame, symbol: str) -> bool:
        """Perform final data quality checks"""
        try:
            # Check for required columns
            required_cols = ['close', 'volume', 'rsi', 'macd', 'atr', 'adx', 'ma7', 'ma25']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"{symbol}: Missing required columns: {missing_cols}")
                return False
                
            # Check for infinite values
            inf_count = np.isinf(df[required_cols]).sum().sum()
            if inf_count > 0:
                self.logger.error(f"{symbol}: Found {inf_count} infinite values")
                return False
                
            # Check for recent missing values
            recent_missing = df[required_cols].iloc[-5:].isnull().sum().sum()
            if recent_missing > 0:
                self.logger.error(f"{symbol}: Found {recent_missing} missing values in recent data")
                return False
                
            # Validate indicator ranges
            last_row = df.iloc[-1]
            
            if not (0 <= last_row['rsi'] <= 100):
                self.logger.error(f"{symbol}: Invalid RSI value: {last_row['rsi']}")
                return False
                
            if last_row['adx'] > 100 or last_row['adx'] < 0:
                self.logger.error(f"{symbol}: Invalid ADX value: {last_row['adx']}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"{symbol}: Error in final quality check: {str(e)}")
            return False

    def _log_final_metrics(self, df: pd.DataFrame, symbol: str):
        """Log final metrics with detailed analysis"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate price change
            price_change = (latest['close'] - prev['close']) / prev['close'] * 100
            
            self.logger.info(f"{symbol}: Processing complete. Shape: {df.shape}, Missing values: {df.isnull().sum().sum()}")
            self.logger.info(f"{symbol} Latest metrics:")
            self.logger.info(f"  Price: {latest['close']:.4f} ({price_change:+.2f}%)")
            self.logger.info(f"  Volume: {latest['volume']:.2f} ({latest['volume']/latest['volume_ma']:.2f}x avg)")
            self.logger.info(f"  RSI: {latest['rsi']:.2f}")
            self.logger.info(f"  MACD: {latest['macd']:.4f}")
            self.logger.info(f"  ATR: {latest['atr']:.4f}")
            self.logger.info(f"  ADX: {latest['adx']:.2f}")
            
        except Exception as e:
            self.logger.error(f"{symbol}: Error logging final metrics: {str(e)}")
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, 
                    fast_period: int = 12, 
                    slow_period: int = 26, 
                    signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        
        plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
        minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
        
        # Calculate TR
        tr = self._calculate_atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def get_last_trading_signals(self) -> Dict[str, TradeSignal]:
        """Retrieve the last generated trading signals"""
        return self._last_trading_signals