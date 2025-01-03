# trade_continious.py

import logging
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, List, Tuple
import time
from binance.client import Client
import hmac
import hashlib
import requests
from binance.exceptions import BinanceAPIException
from config import Config
from utils.logging_setup import setup_logging
from utils.binance_setup import setup_binance_client
from services.trade_analyzer import TradeAnalyzer, TradeSignal
import math

class TradeAllocator:
    """Enhanced trade allocation based on market conditions and signal strength"""
    
    def __init__(self, total_amount: float, min_trade: float = 15, max_trade: float = 25):
        self.total_amount = total_amount
        self.min_trade = min_trade
        self.max_trade = max_trade
        self.logger = logging.getLogger(__name__)
        
    def calculate_score(self, signal: TradeSignal) -> float:
        """Calculate enhanced allocation score based on technical patterns"""
        try:
            # Volume score (higher weight due to strong correlation with success)
            volume_condition = next(c for name, c in signal.market_conditions.items() 
                                  if 'volume' in name.lower())
            volume_score = volume_condition.value / volume_condition.threshold

            # Volatility score (inverse relationship - lower is better)
            volatility_condition = next(c for name, c in signal.market_conditions.items() 
                                      if 'volatility' in name.lower())
            volatility_score = max(0, 1 - (volatility_condition.value / volatility_condition.threshold))

            # RSI score (highest near 50)
            rsi_condition = next(c for name, c in signal.market_conditions.items() 
                               if 'rsi' in name.lower())
            rsi_center = 50
            rsi_score = 1 - abs(rsi_condition.value - rsi_center) / 50

            # Technical conditions combined score
            conditions_met = sum(1 for cond in signal.market_conditions.values() if cond.met)
            condition_score = conditions_met / len(signal.market_conditions)

            # Risk/reward consideration
            risk_reward_score = min(1.0, signal.risk_reward_ratio / 3.0)

            # Weighted final score
            score = (
                0.30 * signal.prediction +      # Model prediction
                0.25 * volume_score +           # Volume impact
                0.15 * condition_score +        # Overall conditions
                0.15 * risk_reward_score +      # Risk/reward ratio
                0.10 * volatility_score +       # Volatility impact
                0.05 * rsi_score               # RSI optimization
            )
            
            return score

        except Exception as e:
            self.logger.error(f"Error calculating score: {str(e)}")
            return 0.0
        
    def allocate_amounts(self, signals: Dict[str, TradeSignal]) -> Dict[str, float]:
        """Allocate trading amounts based on enhanced scoring"""
        try:
            # Filter valid signals and calculate scores
            valid_trades: List[Tuple[str, TradeSignal, float]] = []
            
            for symbol, signal in signals.items():
                if signal and signal.signal_found and signal.prediction > 0.8:
                    score = self.calculate_score(signal)
                    if score > 0.7:  # Minimum score threshold
                        valid_trades.append((symbol, signal, score))
                    
            if not valid_trades:
                return {}
                
            # Sort by score
            valid_trades.sort(key=lambda x: x[2], reverse=True)
            
            # Calculate allocations
            total_score = sum(score for _, _, score in valid_trades)
            allocations = {}
            remaining_amount = self.total_amount
            
            for symbol, signal, score in valid_trades:
                if remaining_amount <= self.min_trade:
                    break
                    
                # Calculate allocation based on score
                base_allocation = (score / total_score) * self.total_amount
                
                # Adjust based on risk/reward
                risk_factor = min(1.2, max(0.8, signal.risk_reward_ratio / 2))
                adjusted_allocation = base_allocation * risk_factor
                
                # Apply constraints
                final_allocation = min(
                    max(adjusted_allocation, self.min_trade),
                    min(self.max_trade, remaining_amount)
                )
                
                if final_allocation >= self.min_trade:
                    allocations[symbol] = final_allocation
                    remaining_amount -= final_allocation
                    
            return allocations

        except Exception as e:
            self.logger.error(f"Error allocating amounts: {str(e)}")
            return {}

class TradeExecutor:
    """Enhanced trade execution with fixed risk/reward OCO orders"""
        
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.open_positions = {}
        
        # Define fixed risk/reward parameters
        self.TAKE_PROFIT_PCT = 0.008  
        self.STOP_LOSS_PCT = 0.005   
    
    def _wait_for_balance_update(self, symbol: str, expected_qty: float, max_attempts: int = 5) -> tuple[float, bool]:
        """Wait for balance to update after market buy with retries"""
        asset = symbol.replace('USDT', '')
        self.logger.info(f"Waiting for {asset} balance to update...")
        
        for attempt in range(max_attempts):
            time.sleep(2)  # Wait 2 seconds between checks
            
            try:
                account = self.client.get_account()
                asset_balance = float(next(
                    bal['free'] for bal in account['balances'] 
                    if bal['asset'] == asset
                ))
                
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts}: {asset} balance = {asset_balance}")
                
                if asset_balance >= expected_qty * 0.99:  # Allow for 1% difference
                    self.logger.info(f"✅ {asset} balance updated successfully")
                    return asset_balance, True
                    
            except Exception as e:
                self.logger.error(f"Error checking balance: {str(e)}")
                
        # Return the last known balance even if we didn't reach expected quantity
        return asset_balance if 'asset_balance' in locals() else 0.0, False
                    
    def execute_trade(self, symbol: str, signal: TradeSignal, usdt_amount: float) -> bool:
        """Execute trade with fixed risk/reward parameters"""
        filled_qty = None
        
        try:
            # Check for existing position
            if symbol in self.open_positions:
                self.logger.warning(f"Already have an open position for {symbol}")
                return False

            # Verify USDT balance
            account = self.client.get_account()
            usdt_balance = float(next(
                asset['free'] for asset in account['balances'] 
                if asset['asset'] == 'USDT'
            ))
            
            if usdt_balance < usdt_amount * 1.01:
                self.logger.error(f"Insufficient USDT balance: {usdt_balance:.2f}")
                return False

            # Get symbol info and current price
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False

            ticker = self.client.get_symbol_ticker(symbol=symbol)
            market_price = float(ticker['price'])
            
            # Calculate fixed take profit and stop loss prices
            take_profit_price = market_price * (1 + self.TAKE_PROFIT_PCT)
            stop_price = market_price * (1 - self.STOP_LOSS_PCT)
            limit_price = stop_price * 0.999  # Slightly below stop for better fills
            
            # Format prices according to symbol precision
            price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', 
                                    symbol_info['filters']))
            price_precision = len(str(float(price_filter['tickSize'])).rstrip('0').split('.')[-1])
            
            take_profit_price = f"{take_profit_price:.{price_precision}f}"
            stop_price = f"{stop_price:.{price_precision}f}"
            limit_price = f"{limit_price:.{price_precision}f}"

            # Calculate and validate quantity
            quantity = self.calculate_quantity(symbol, market_price, usdt_amount)
            if not quantity:
                return False
            
            # Log trade setup
            self.logger.info(f"\n{symbol} Trade Setup:")
            self.logger.info(f"  USDT Amount: {usdt_amount:.2f}")
            self.logger.info(f"  Market Price: {market_price:.8f}")
            self.logger.info(f"  Quantity: {quantity}")
            self.logger.info(f"  Take Profit (+0.8%): {take_profit_price}")
            self.logger.info(f"  Stop Loss (-0.5%): {stop_price}")
            self.logger.info(f"  Limit Price: {limit_price}")

            # Place market buy order
            try:
                market_order = self.client.create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=f"{quantity:.8f}".rstrip('0').rstrip('.'),
                    newOrderRespType='FULL'
                )
                
                if market_order['status'] != 'FILLED':
                    self.logger.error(f"Market buy not filled: {market_order}")
                    return False
                    
                filled_qty = float(market_order['executedQty'])
                filled_price = float(market_order['fills'][0]['price'])
                self.logger.info(f"Market buy filled: {filled_qty} @ {filled_price}")

                # Track position
                self.open_positions[symbol] = {
                    'quantity': filled_qty,
                    'entry_price': filled_price,
                    'time': time.time()
                }
                
                # Wait for balance update
                actual_balance, balance_updated = self._wait_for_balance_update(symbol, filled_qty)
                
                if not balance_updated:
                    self.logger.warning(f"Balance update timeout. Using filled quantity: {filled_qty}")
                    adjusted_qty = filled_qty * 0.9999  # Slightly reduce quantity
                    
                    # Try OCO with adjusted quantity
                    oco_result = self.place_oco_order(
                        symbol=symbol,
                        side='SELL',
                        quantity=adjusted_qty,
                        stop_price=stop_price,
                        limit_price=limit_price,
                        take_profit_price=take_profit_price
                    )
                    
                    if oco_result:
                        self.logger.info(f"✅ OCO order placed with adjusted quantity: {adjusted_qty}")
                        return True
                    else:
                        self.logger.error(f"❌ OCO placement failed for {symbol}")
                        return self._close_position(symbol, adjusted_qty)
                
                # Use actual balance for OCO
                oco_result = self.place_oco_order(
                    symbol=symbol,
                    side='SELL',
                    quantity=actual_balance,
                    stop_price=stop_price,
                    limit_price=limit_price,
                    take_profit_price=take_profit_price
                )

                if oco_result:
                    self.logger.info(f"✅ Trade successful: {symbol}")
                    return True
                else:
                    self.logger.error(f"❌ OCO order failed for {symbol}")
                    return self._close_position(symbol, actual_balance)

            except BinanceAPIException as e:
                self.logger.error(f"Market buy failed: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            if filled_qty:
                self.open_positions.pop(symbol, None)
                return self._close_position(symbol, filled_qty)
            return False


    def _handle_oco_failure(self, symbol: str, quantity: float) -> None:
        """Handle OCO order failure by attempting market close"""
        try:
            self.logger.warning(f"Attempting to close position with market order for {symbol}")
            close_order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            if close_order['status'] == 'FILLED':
                self.logger.info(f"Position closed successfully for {symbol}")
            else:
                self.logger.error(f"Failed to close position: {close_order}")
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")

    def _log_trade_setup(self, symbol: str, amount: float, price: float, 
                        quantity: float, target: str, stop: str, limit: str) -> None:
        """Log trade setup details"""
        self.logger.info(f"\n{symbol} Trade Setup:")
        self.logger.info(f"  USDT Amount: {amount:.2f}")
        self.logger.info(f"  Market Price: {price:.8f}")
        self.logger.info(f"  Quantity: {quantity}")
        self.logger.info(f"  Target Price: {target}")
        self.logger.info(f"  Stop Price: {stop}")
        self.logger.info(f"  Limit Price: {limit}")

    def place_oco_order(self, symbol: str, side: str, quantity: float, 
                       stop_price: str, limit_price: str, take_profit_price: str) -> Optional[dict]:
        """Place OCO order with proper formatting and error handling"""
        try:
            # Format quantity with proper precision
            symbol_info = self.get_symbol_info(symbol)
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                                 symbol_info['filters']))
            step_size = float(lot_size['stepSize'])
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            formatted_qty = f"{quantity:.{precision}f}".rstrip('0').rstrip('.')

            # OCO order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'quantity': formatted_qty,
                'price': take_profit_price,         # Take profit price
                'stopPrice': stop_price,            # Stop trigger price
                'stopLimitPrice': limit_price,      # Stop limit price
                'stopLimitTimeInForce': 'GTC',
                'timestamp': str(int(time.time() * 1000)),
                'recvWindow': '5000'
            }

            # Generate query string
            query_string = '&'.join([f"{key}={params[key]}" for key in params.keys()])
            
            # Generate signature
            signature = hmac.new(
                self.client.API_SECRET.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature and make request
            params['signature'] = signature
            headers = {'X-MBX-APIKEY': self.client.API_KEY}
            url = 'https://api.binance.com/api/v3/order/oco'

            self.logger.info(f"Sending OCO order for {symbol}:")
            self.logger.info(f"  Quantity: {formatted_qty}")
            self.logger.info(f"  Take Profit: {take_profit_price}")
            self.logger.info(f"  Stop Price: {stop_price}")
            self.logger.info(f"  Limit Price: {limit_price}")

            response = requests.post(url, headers=headers, params=params)
            
            if response.status_code == 200:
                order_data = response.json()
                self.logger.info(f"OCO order successfully placed for {symbol}")
                self._log_order_response(order_data)
                return order_data
            else:
                self.logger.error(f"OCO order failed. Status: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error placing OCO order: {str(e)}")
            return None

    def _close_position(self, symbol: str, quantity: float) -> bool:
        """Close position with market order and update tracking"""
        try:
            self.logger.warning(f"Attempting to close position with market order for {symbol}")
            
            # Verify we have the asset balance
            account = self.client.get_account()
            asset = symbol.replace('USDT', '')
            asset_balance = float(next(
                bal['free'] for bal in account['balances'] 
                if bal['asset'] == asset
            ))
            
            if asset_balance < quantity:
                self.logger.error(f"Insufficient {asset} balance for closing trade")
                return False

            # Format quantity with proper precision
            symbol_info = self.get_symbol_info(symbol)
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                            symbol_info['filters']))
            step_size = float(lot_size['stepSize'])
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            formatted_qty = f"{quantity:.{precision}f}"

            # Place market sell order
            close_order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=formatted_qty,
                newOrderRespType='FULL'
            )
            
            if close_order['status'] == 'FILLED':
                filled_price = float(close_order['fills'][0]['price'])
                self.logger.info(f"Position closed at {filled_price} for {symbol}")
                # Remove from open positions
                self.open_positions.pop(symbol, None)
                return True
            else:
                self.logger.error(f"Failed to close position: {close_order}")
                return False
                
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Error closing position: {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False
    
    def validate_basic_prices(self, symbol: str, market_price: float,
                                target_price: float, stop_price: float) -> bool:
            """Basic price validation without strict R/R requirements"""
            try:
                # Validate target is above market for longs
                if target_price <= market_price:
                    self.logger.warning(f"{symbol}: Target price {target_price} must be above market price {market_price}")
                    return False
                    
                # Validate stop is below market for longs
                if stop_price >= market_price:
                    self.logger.warning(f"{symbol}: Stop price {stop_price} must be below market price {market_price}")
                    return False
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error validating prices: {e}")
                return False

    def check_min_notional(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if order meets minimum notional value requirements"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False
            
            # Find MIN_NOTIONAL filter
            min_notional_filter = None
            for filter_item in symbol_info['filters']:
                if filter_item['filterType'] == 'NOTIONAL':
                    min_notional_filter = filter_item
                    break
                elif filter_item['filterType'] == 'MIN_NOTIONAL':
                    min_notional_filter = filter_item
                    break
            
            if not min_notional_filter:
                self.logger.warning(f"{symbol}: Could not find MIN_NOTIONAL filter")
                return True  # Continue if filter not found
            
            # Get minimum notional value
            min_notional = float(min_notional_filter.get('minNotional', 0))
            if 'minNotional' not in min_notional_filter:
                min_notional = float(min_notional_filter.get('notional', 0))
            
            # Calculate order notional value
            notional = quantity * price
            
            if notional < min_notional:
                self.logger.warning(
                    f"{symbol}: Order notional {notional:.2f} USDT below minimum {min_notional} USDT"
                )
                return False
            
            self.logger.info(f"{symbol}: Notional value check passed: {notional:.2f} USDT")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking min notional: {str(e)}")
            return False  # Return False on error to be safe

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading info"""
        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                raise ValueError(f"No symbol info found for {symbol}")
            return info
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def calculate_quantity(self, symbol: str, entry_price: float, usdt_amount: float) -> Optional[float]:
        """Calculate quantity with precision handling"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
                
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                                symbol_info['filters']))
            min_qty = float(lot_size['minQty'])
            max_qty = float(lot_size.get('maxQty', float('inf')))
            step_size = float(lot_size['stepSize'])
            
            quantity = usdt_amount / entry_price
            
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            quantity = round(quantity / step_size) * step_size
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal(str(step_size)), 
                rounding=ROUND_DOWN
            ))
            
            quantity = max(min(quantity, max_qty), min_qty)
            
            self.logger.info(f"{symbol}: Calculated quantity: {quantity}")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity for {symbol}: {e}")
            return None

    def _format_price(self, price: float, symbol_info: Dict) -> str:
        """Format price according to symbol's price filter"""
        try:
            price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', 
                                     symbol_info['filters']))
            tick_size = float(price_filter['tickSize'])
            precision = len(str(tick_size).rstrip('0').split('.')[-1])
            return f"{price:.{precision}f}"
        except Exception as e:
            self.logger.error(f"Error formatting price: {e}")
            return str(price)

    def _handle_binance_error(self, e: BinanceAPIException, order_type: str) -> None:
        """Handle specific Binance API errors"""
        error_messages = {
            -1013: "Filter failure (e.g., PRICE_FILTER, LOT_SIZE)",
            -1021: "Timestamp outside of recv_window",
            -2010: "New order rejected (insufficient balance)",
            -2011: "Cancel order rejected (unknown order)",
            -1102: "Mandatory parameter missing",
        }
        
        error_code = int(e.code)
        error_msg = error_messages.get(error_code, e.message)
        self.logger.error(f"{order_type} order failed: {error_code} - {error_msg}")

    def _log_order_response(self, response_data: Dict) -> None:
        """Log order response details"""
        try:
            self.logger.info("Order Response:")
            self.logger.info(f"  Order List ID: {response_data.get('orderListId')}")
            self.logger.info(f"  Status: {response_data.get('listStatusType')}")
            
            for order in response_data.get('orders', []):
                self.logger.info(f"\n  Order Details:")
                self.logger.info(f"    Symbol: {order.get('symbol')}")
                self.logger.info(f"    Order ID: {order.get('orderId')}")
                self.logger.info(f"    Client Order ID: {order.get('clientOrderId')}")
            
            for report in response_data.get('orderReports', []):
                self.logger.info(f"\n  Order Report:")
                self.logger.info(f"    Order ID: {report.get('orderId')}")
                self.logger.info(f"    Side: {report.get('side')}")
                self.logger.info(f"    Type: {report.get('type')}")
                self.logger.info(f"    Price: {report.get('price')}")
                if 'stopPrice' in report:
                    self.logger.info(f"    Stop Price: {report.get('stopPrice')}")
                self.logger.info(f"    Status: {report.get('status')}")
                
        except Exception as e:
            self.logger.error(f"Error logging order response: {e}")

class PositionMonitor:
    """Monitor and manage positions that don't have corresponding sell orders"""
    
    def __init__(self, client: Client, executor: TradeExecutor):
        self.client = client
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
    def get_open_positions(self) -> Dict[str, float]:
        """Get all positions without sell orders"""
        try:
            # Get account information
            account = self.client.get_account()
            
            # Filter for non-zero balances (excluding USDT)
            positions = {}
            for balance in account['balances']:
                asset = balance['asset']
                free_amount = float(balance['free'])
                
                if asset != 'USDT' and free_amount > 0:
                    symbol = f"{asset}USDT"
                    
                    # Check if symbol is valid
                    try:
                        symbol_info = self.client.get_symbol_info(symbol)
                        if symbol_info and symbol_info['status'] == 'TRADING':
                            positions[symbol] = free_amount
                    except Exception as e:
                        self.logger.warning(f"Error checking symbol {symbol}: {str(e)}")
                        continue
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {str(e)}")
            return {}
            
    def check_existing_orders(self, symbol: str) -> bool:
        """Check if there are any open orders for the symbol"""
        try:
            open_orders = self.client.get_open_orders(symbol=symbol)
            return len(open_orders) > 0
        except Exception as e:
            self.logger.error(f"Error checking orders for {symbol}: {str(e)}")
            return False
            
    def place_limit_sell_order(self, symbol: str, quantity: float, price: str) -> bool:
        """Place a simple limit sell order"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )
            self.logger.info(f"Limit sell order placed successfully for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error placing limit sell order for {symbol}: {str(e)}")
            return False

    def adjust_quantity(self, symbol: str, quantity: float, reduction_pct: float) -> str:
        """Adjust quantity with proper precision"""
        try:
            symbol_info = self.executor.get_symbol_info(symbol)
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                                symbol_info['filters']))
            step_size = float(lot_size['stepSize'])
            qty_precision = len(str(step_size).rstrip('0').split('.')[-1])
            
            # Reduce quantity by percentage
            adjusted_qty = quantity * (1 - reduction_pct)
            
            # Round to step size
            adjusted_qty = math.floor(adjusted_qty / step_size) * step_size
            
            # Format with proper precision
            formatted_qty = f"{adjusted_qty:.{qty_precision}f}".rstrip('0').rstrip('.')
            return formatted_qty
            
        except Exception as e:
            self.logger.error(f"Error adjusting quantity: {str(e)}")
            return str(quantity)

    def try_place_oco_order(self, symbol: str, quantity: float, current_price: float,
                           formatted_tp: str, formatted_stop: str, formatted_limit: str,
                           max_retries: int = 3) -> bool:
        """Try to place OCO order with retries and quantity adjustments"""
        original_qty = quantity
        
        for attempt in range(max_retries):
            try:
                # Progressive reduction for each retry
                if attempt > 0:
                    # Increase reduction percentage with each retry
                    reduction = 0.002 * attempt  # Start with 0.2% reduction
                    adjusted_qty = self.adjust_quantity(symbol, original_qty, reduction)
                    self.logger.info(f"Retry {attempt + 1} with adjusted quantity: {adjusted_qty} (reduced by {reduction:.2%})")
                else:
                    # First attempt with 0.1% reduction for safety
                    adjusted_qty = self.adjust_quantity(symbol, original_qty, 0.001)
                    self.logger.info(f"Initial attempt with quantity: {adjusted_qty} (reduced by 0.1%)")
                
                # Additional safety check for minimum quantity
                symbol_info = self.executor.get_symbol_info(symbol)
                lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                                    symbol_info['filters']))
                min_qty = float(lot_size['minQty'])
                
                if float(adjusted_qty) < min_qty:
                    self.logger.warning(f"Adjusted quantity {adjusted_qty} below minimum {min_qty}")
                    continue
                
                self.logger.info(f"OCO order attempt {attempt + 1}/{max_retries} for {symbol}")
                
                oco_result = self.executor.place_oco_order(
                    symbol=symbol,
                    side='SELL',
                    quantity=float(adjusted_qty),
                    stop_price=formatted_stop,
                    limit_price=formatted_limit,
                    take_profit_price=formatted_tp
                )
                
                if oco_result:
                    self.logger.info(f"✅ Successfully placed OCO order for {symbol}")
                    return True
                    
            except BinanceAPIException as e:
                if e.code == -2010:  # Insufficient balance
                    self.logger.warning(f"Insufficient balance error, reducing quantity further")
                    continue
                self.logger.error(f"OCO order attempt {attempt + 1} failed: {str(e)}")
            except Exception as e:
                self.logger.error(f"OCO order attempt {attempt + 1} failed: {str(e)}")
            
            time.sleep(1)  # Wait before retry
            
        return False

    def monitor_positions(self) -> None:
        """Monitor positions and place OCO orders where needed"""
        try:
            # Get all positions
            positions = self.get_open_positions()
            
            if not positions:
                self.logger.info("No open positions found")
                return
                
            self.logger.info(f"Found {len(positions)} positions to check")
            
            for symbol, quantity in positions.items():
                try:
                    # Skip if already has open orders
                    if self.check_existing_orders(symbol):
                        self.logger.info(f"{symbol} already has open orders")
                        continue
                        
                    # Get current market price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calculate OCO prices
                    take_profit_price = current_price * (1 + self.executor.TAKE_PROFIT_PCT)
                    stop_price = current_price * (1 - self.executor.STOP_LOSS_PCT)
                    limit_price = stop_price * 0.999
                    
                    # Format prices according to symbol precision
                    symbol_info = self.executor.get_symbol_info(symbol)
                    price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', 
                                            symbol_info['filters']))
                    price_precision = len(str(float(price_filter['tickSize'])).rstrip('0').split('.')[-1])
                    
                    formatted_tp = f"{take_profit_price:.{price_precision}f}"
                    formatted_stop = f"{stop_price:.{price_precision}f}"
                    formatted_limit = f"{limit_price:.{price_precision}f}"
                    
                    # Format quantity with proper precision
                    lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                                        symbol_info['filters']))
                    step_size = float(lot_size['stepSize'])
                    qty_precision = len(str(step_size).rstrip('0').split('.')[-1])
                    formatted_qty = f"{quantity:.{qty_precision}f}".rstrip('0').rstrip('.')
                    
                    # Validate minimum notional
                    if not self.executor.check_min_notional(symbol, float(formatted_qty), current_price):
                        self.logger.warning(f"Position for {symbol} too small for OCO order")
                        continue
                    
                    # Try OCO order first
                    self.logger.info(f"Placing OCO order for {symbol}:")
                    self.logger.info(f"  Quantity: {formatted_qty}")
                    self.logger.info(f"  Current Price: {current_price}")
                    self.logger.info(f"  Take Profit: {formatted_tp}")
                    self.logger.info(f"  Stop Price: {formatted_stop}")
                    
                    oco_success = self.try_place_oco_order(
                        symbol=symbol,
                        quantity=float(formatted_qty),
                        current_price=current_price,
                        formatted_tp=formatted_tp,
                        formatted_stop=formatted_stop,
                        formatted_limit=formatted_limit
                    )
                    
                    # If OCO fails, try simple limit sell with much smaller quantity
                    if not oco_success:
                        self.logger.warning(f"OCO order failed for {symbol}, attempting limit sell order with reduced quantity")
                        
                        # Try with 0.1% of the original quantity
                        minimal_qty = self.adjust_quantity(symbol, float(formatted_qty), 0.999)
                        self.logger.info(f"Attempting minimal quantity sale: {minimal_qty}")
                        
                        limit_success = self.place_limit_sell_order(
                            symbol=symbol,
                            quantity=float(minimal_qty),
                            price=formatted_tp
                        )
                        
                        if limit_success:
                            self.logger.info(f"✅ Successfully placed minimal limit sell order for {symbol}")
                        else:
                            self.logger.error(f"❌ Failed to place any sell orders for {symbol} even with minimal quantity")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
                    
                time.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
    
    def try_place_limit_sell(self, symbol: str, quantity: float, price: str, 
                         max_retries: int = 2) -> bool:
        """Try to place limit sell order with retries and minimal quantity"""
        original_qty = quantity
        
        for attempt in range(max_retries):
            try:
                # Use 0.1% of original quantity for final fallback
                reduction = 0.999 if attempt == 0 else 0.9995
                minimal_qty = self.adjust_quantity(symbol, original_qty, reduction)
                
                self.logger.info(f"Attempting limit sell with {minimal_qty} ({reduction:.4%} reduction)")
                
                # Validate minimum notional
                if not self.executor.check_min_notional(symbol, float(minimal_qty), float(price)):
                    self.logger.warning(f"Quantity too small for minimum notional")
                    continue
                
                order = self.client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=minimal_qty,
                    price=price
                )
                
                self.logger.info(f"✅ Limit sell order placed successfully for {symbol}")
                return True
                
            except BinanceAPIException as e:
                if e.code == -2010:  # Insufficient balance
                    continue
                self.logger.error(f"Limit order attempt {attempt + 1} failed: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error placing limit order: {str(e)}")
                
            time.sleep(1)
        
        return False

def check_balance(client: Client) -> float:
    """Check current USDT balance"""
    logger = logging.getLogger(__name__)
    
    account = client.get_account()
    usdt_balance = float(next(
        asset['free'] for asset in account['balances'] 
        if asset['asset'] == 'USDT'
    ))
    logger.info(f"Current USDT balance: {usdt_balance:.2f}")
    return usdt_balance

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a more compact and clean progress bar"""
    progress = float(current) / float(total)
    filled = int(width * progress)
    bar = '▓' * filled + '░' * (width - filled)
    percent = int(progress * 100)
    return f"{percent:3d}% [{bar}]"

def loading_wait(seconds: int, message: str) -> None:
    """More concise and informative waiting method"""
    logger = logging.getLogger(__name__)
    
    for elapsed in range(seconds):
        remaining = seconds - elapsed
        progress_bar = create_progress_bar(elapsed + 1, seconds)
        
        # Use print instead of logger.info for cleaner output
        print(f"\r{message} - {progress_bar} | Time Left: {remaining:3d}s", end='', flush=True)
        time.sleep(1)
    
    # Clear the line
    print('\r' + ' ' * 100 + '\r', end='', flush=True)

def execute_trades(client: Client, analyzer: TradeAnalyzer, 
                  executor: TradeExecutor, position_monitor: PositionMonitor, last_check_time: float) -> None:
    """Execute trades when opportunities are found"""
    logger = logging.getLogger(__name__)
    
    try:
        # Execute trades using previously generated signals
        executed_trades = 0
        total_invested = 0
        failed_trades = []
        
        account = client.get_account()
        available_balance = float(next(
            asset['free'] for asset in account['balances'] 
            if asset['asset'] == 'USDT'
        ))
        
        # IMPORTANT: Use the signals passed to the function, do NOT regenerate
        signals = analyzer.get_last_trading_signals()
        
        # More verbose logging about signals
        logger.info(f"Total signals generated: {len(signals)}")
        for symbol, signal in signals.items():
            logger.info(f"Signal details for {symbol}:")
            logger.info(f"  Signal Found: {signal.signal_found}")
            logger.info(f"  Prediction: {signal.prediction}")
            logger.info(f"  Current Price: {signal.current_price}")
        
        allocator = TradeAllocator(available_balance * 0.99)  # 1% buffer
        allocations = allocator.allocate_amounts(signals)
        
        # More verbose logging about allocations
        logger.info(f"Total allocations: {len(allocations)}")
        for symbol, amount in allocations.items():
            logger.info(f"Allocation for {symbol}: {amount:.2f} USDT")
        
        for symbol, amount in allocations.items():
            # Skip if we already have an open position
            if symbol in executor.open_positions:
                logger.info(f"Skipping {symbol} - existing position")
                continue
                
            # Skip if there are existing orders
            if position_monitor.check_existing_orders(symbol):
                logger.info(f"Skipping {symbol} - has open orders")
                continue
            
            if available_balance < amount * 1.01:
                logger.warning(f"Insufficient balance for {symbol}. Required: {amount:.2f}, Available: {available_balance:.2f}")
                failed_trades.append((symbol, "Insufficient balance"))
                continue
            
            signal = signals[symbol]
            logger.info(f"\nExecuting trade for {symbol}...")
            
            try:
                # Check price drift
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                price_drift = abs(current_price - signal.current_price) / signal.current_price
                
                if price_drift > 0.005:
                    logger.warning(f"Price drifted {price_drift:.2%}. Skipping {symbol}")
                    failed_trades.append((symbol, "Price drift too high"))
                    continue
                
                success = executor.execute_trade(symbol, signal, amount)
                
                if success:
                    executed_trades += 1
                    total_invested += amount
                    available_balance -= amount
                    logger.info(f"✅ Trade successful: {symbol}")
                else:
                    failed_trades.append((symbol, "Execution failed"))
                    logger.error(f"❌ Trade failed: {symbol}")
                    
            except Exception as e:
                logger.error(f"Error trading {symbol}: {str(e)}")
                failed_trades.append((symbol, f"Error: {str(e)}"))
            
            time.sleep(0.1)  # Rate limiting
        
        # Print execution summary
        if executed_trades > 0 or failed_trades:
            logger.info("\n=== Trade Summary ===")
            logger.info(f"Trades: {executed_trades}/{len(allocations)} successful")
            logger.info(f"Invested: {total_invested:.2f} USDT")
            logger.info(f"Remaining: {available_balance:.2f} USDT")
            
            if failed_trades:
                logger.info("\nFailed Trades:")
                for symbol, reason in failed_trades:
                    logger.info(f"  {symbol}: {reason}")
        
    except Exception as e:
        logger.error(f"Error in trade execution: {str(e)}")
        raise
    
def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = Config()
    
    try:
        # Initialize components once outside the main loop
        client = setup_binance_client()
        client.ping()
        
        analyzer = TradeAnalyzer(client, config)
        executor = TradeExecutor(client)
        position_monitor = PositionMonitor(client, executor)
        
        if not analyzer.initialize_traders():
            logger.error("Failed to initialize traders. Exiting.")
            return
        
        last_trade_time = 0
        TRADE_COOLDOWN_PERIOD = 15 * 60 
        
        while True:
            try:
                current_time = time.time()
                
                # Check Balance
                logger.info("\n=== Checking Balance ===")
                usdt_balance = check_balance(client)
                
                # Trading Logic
                if (usdt_balance >= 15 and 
                    current_time - last_trade_time >= TRADE_COOLDOWN_PERIOD):
                    
                    logger.info("\n=== Looking for Trading Opportunities ===")
                    signals = analyzer.generate_trading_signals()
                    
                    logger.info(f"Total signals generated: {len(signals)}")
                    
                    if signals:
                        logger.info("Trading signals found, calculating allocations...")
                        available_balance = usdt_balance * 0.99
                        allocator = TradeAllocator(available_balance)
                        allocations = allocator.allocate_amounts(signals)
                        
                        logger.info(f"Total allocations: {len(allocations)}")
                        for symbol, amount in allocations.items():
                            logger.info(f"Allocation for {symbol}: {amount:.2f} USDT")
                        
                        if allocations:
                            logger.info("Executing trades...")
                            execute_trades(client, analyzer, executor, position_monitor, current_time)
                            
                            # Update last trade time after successful trade execution
                            last_trade_time = current_time
                            
                            logger.info(f"Next trades possible after: {time.ctime(last_trade_time + TRADE_COOLDOWN_PERIOD)}")
                        else:
                            logger.warning("No trade allocations were made")
                else:
                    if current_time - last_trade_time < TRADE_COOLDOWN_PERIOD:
                        remaining_cooldown = int(TRADE_COOLDOWN_PERIOD - (current_time - last_trade_time))
                        logger.info(f"In trade cooldown. {remaining_cooldown} seconds remaining.")
                
                # Check and Monitor Positions
                logger.info("\n=== Checking Open Positions ===")
                positions = position_monitor.get_open_positions()
                
                if positions:
                    logger.info(f"Found {len(positions)} positions to check")
                    position_monitor.monitor_positions()
                
                # Wait before next iteration
                time.sleep(60)  
                
            except BinanceAPIException as e:
                logger.error(f"Binance API Error: {e.code} - {e.message}")
                time.sleep(60)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error: {str(e)}")
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                logger.exception(e)
                time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Trading bot shutting down...")
    
    except Exception as e:
        logger.error(f"Critical initialization error: {str(e)}")

if __name__ == "__main__":
    main()