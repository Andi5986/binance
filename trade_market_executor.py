# trade_market_executor.py

import logging
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, List, Tuple
import time
import numpy as np
from binance.client import Client
import hmac
import hashlib
import requests
from binance.exceptions import BinanceAPIException
from config import Config
from utils.logging_setup import setup_logging
from utils.binance_setup import setup_binance_client
from services.trade_analyzer import TradeAnalyzer, TradeSignal
import os

class TradeAllocator:
    """Enhanced trade allocation based on market conditions and signal strength"""
    
    def __init__(self, total_amount: float, min_trade: float = 15, max_trade: float = 100):
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
    """Enhanced trade execution with market entry and OCO exit orders"""
        
    def __init__(self, client: Client):
        """Initialize TradeExecutor with Binance client"""
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.open_positions = {} 
    
    def _wait_for_balance_update(self, symbol: str, expected_qty: float, max_attempts: int = 5) -> bool:
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
                
                if asset_balance >= expected_qty * 0.99999:  # Allow for minimal rounding differences
                    self.logger.info(f"✅ {asset} balance updated successfully")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Error checking balance: {str(e)}")
                
        self.logger.error(f"❌ Timed out waiting for {asset} balance update")
        return False
                    
    def execute_trade(self, symbol: str, signal: TradeSignal, usdt_amount: float) -> bool:
        """Execute trade with better order flow handling"""
        filled_qty = None
        
        try:
            # Check if we already have an open position
            if symbol in self.open_positions:
                self.logger.warning(f"Already have an open position for {symbol}")
                return False

            # Verify available balance with buffer
            account = self.client.get_account()
            usdt_balance = float(next(
                asset['free'] for asset in account['balances'] 
                if asset['asset'] == 'USDT'
            ))
            
            if usdt_balance < usdt_amount * 1.01:  # 1% buffer
                self.logger.error(f"Insufficient USDT balance. Required: {usdt_amount:.2f}, Available: {usdt_balance:.2f}")
                return False

            # Get symbol info and market price
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False

            # Format prices with proper precision
            price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', 
                                    symbol_info['filters']))
            price_precision = len(str(float(price_filter['tickSize'])).rstrip('0').split('.')[-1])

            ticker = self.client.get_symbol_ticker(symbol=symbol)
            market_price = float(ticker['price'])
            
            target_price = f"{signal.target_price:.{price_precision}f}"
            stop_price = f"{signal.stop_price:.{price_precision}f}"
            limit_price = f"{signal.stop_price * 0.999:.{price_precision}f}"

            # Calculate quantity
            quantity = self.calculate_quantity(symbol, market_price, usdt_amount)
            if not quantity:
                return False
            
            # Log trade setup
            self.logger.info(f"\n{symbol} Trade Setup:")
            self.logger.info(f"  USDT Amount: {usdt_amount:.2f}")
            self.logger.info(f"  Market Price: {market_price:.8f}")
            self.logger.info(f"  Quantity: {quantity}")
            self.logger.info(f"  Target Price: {target_price}")
            self.logger.info(f"  Stop Price: {stop_price}")
            self.logger.info(f"  Limit Price: {limit_price}")

            # 1. Place market buy order
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

                # Track the position
                self.open_positions[symbol] = {
                    'quantity': filled_qty,
                    'entry_price': filled_price,
                    'time': time.time()
                }
                
                # Wait for balance to update with retries
                if not self._wait_for_balance_update(symbol, filled_qty):
                    self.logger.error(f"Failed to confirm {symbol} balance update")
                    return self._close_position(symbol, filled_qty)

            except BinanceAPIException as e:
                self.logger.error(f"Market buy failed: {str(e)}")
                return False

            # Place OCO sell order
            if filled_qty:
                # Double check the balance one more time before OCO
                account = self.client.get_account()
                asset = symbol.replace('USDT', '')
                asset_balance = float(next(
                    bal['free'] for bal in account['balances'] 
                    if bal['asset'] == asset
                ))
                
                if asset_balance < filled_qty * 0.99999:  # Allow for minimal rounding differences
                    self.logger.error(f"Insufficient {asset} balance ({asset_balance}) for OCO order. Required: {filled_qty}")
                    return self._close_position(symbol, asset_balance)

                self.logger.info(f"Placing OCO order with confirmed {asset} balance: {asset_balance}")
                oco_result = self.place_oco_order(
                    symbol=symbol,
                    side='SELL',
                    quantity=filled_qty,
                    stop_price=stop_price,
                    limit_price=limit_price,
                    take_profit_price=target_price
                )

                if oco_result:
                    self.logger.info(f"✅ Trade successful: {symbol}")
                    return True
                else:
                    self.logger.error(f"❌ OCO order failed for {symbol}")
                    # Remove from open positions
                    self.open_positions.pop(symbol, None)
                    # Try to close position
                    return self._close_position(symbol, filled_qty)

            return False

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            # Try to close position if we have a filled quantity
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
        """Place an OCO (One-Cancels-Other) order with proper formatting"""
        try:
            # Format quantity to match symbol's precision
            symbol_info = self.get_symbol_info(symbol)
            lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', 
                            symbol_info['filters']))
            step_size = float(lot_size['stepSize'])
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            formatted_qty = f"{quantity:.{precision}f}".rstrip('0').rstrip('.')

            # Parameters for the OCO order - using the standard OCO parameters
            params = {
                'symbol': symbol,
                'side': side,
                'quantity': formatted_qty,
                'price': take_profit_price,         # Limit order price (take profit)
                'stopPrice': stop_price,            # Stop trigger price
                'stopLimitPrice': limit_price,      # Stop limit price
                'stopLimitTimeInForce': 'GTC',
                'timestamp': str(int(time.time() * 1000)),
                'recvWindow': '5000'
            }

            # Generate query string WITHOUT sorting (Binance OCO is order-sensitive)
            query_string = '&'.join([f"{key}={params[key]}" for key in params.keys()])
            
            # Generate signature
            signature = hmac.new(
                self.client.API_SECRET.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature to parameters
            params['signature'] = signature

            # Make request
            headers = {'X-MBX-APIKEY': self.client.API_KEY}
            url = 'https://api.binance.com/api/v3/order/oco'

            self.logger.info(f"Sending OCO order for {symbol}:")
            self.logger.info(f"  Quantity: {formatted_qty}")
            self.logger.info(f"  Take Profit: {take_profit_price}")
            self.logger.info(f"  Stop Price: {stop_price}")
            self.logger.info(f"  Limit Price: {limit_price}")

            response = requests.post(url, headers=headers, params=params)
            
            # Handle response
            if response.status_code == 200:
                order_data = response.json()
                self.logger.info(f"OCO order successfully placed for {symbol}")
                self._log_order_response(order_data)
                return order_data
            else:
                self.logger.error(f"OCO order failed. Status: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                self.logger.error(f"Query string: {query_string}")
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

def execute_trades():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = Config()
    
    try:
        if not Path('.env').exists():
            logger.error("❌ .env file not found")
            return

        client = setup_binance_client()
        client.ping()

        total_amount = float(input("\nEnter total USDT amount to trade: "))
        if total_amount <= 0:
            logger.error("Trade amount must be positive")
            return
            
        analyzer = TradeAnalyzer(client, config)
        executor = TradeExecutor(client)
        allocator = TradeAllocator(total_amount)
        
        logger.info(f"\n=== Starting Trade Execution | Amount: {total_amount:.2f} USDT ===")
        
        account = client.get_account()
        usdt_balance = float(next(
            asset['free'] for asset in account['balances'] 
            if asset['asset'] == 'USDT'
        ))
        
        if usdt_balance < total_amount:
            logger.error(f"❌ Insufficient balance. Available: {usdt_balance:.2f}")
            return
            
        if not analyzer.initialize_traders():
            logger.error("Failed to initialize traders")
            return
            
        signals = analyzer.generate_trading_signals()
        if not signals:
            logger.warning("No valid trading signals found")
            return
            
        allocations = allocator.allocate_amounts(signals)
        if not allocations:
            logger.warning("No valid allocations generated")
            return
        
        # Execute trades immediately
        executed_trades = 0
        total_invested = 0
        failed_trades = []
        
        # Track available balance during execution
        available_balance = usdt_balance
        
        for symbol, amount in allocations.items():
            if available_balance < amount * 1.01:  # Include 1% buffer
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
                
                # Single execution attempt without retries
                success = executor.execute_trade(symbol, signal, amount)
                    
                if success:
                    executed_trades += 1
                    total_invested += amount
                    available_balance -= amount  # Update available balance
                    logger.info(f"✅ Trade successful: {symbol}")
                else:
                    failed_trades.append((symbol, "Execution failed"))
                    logger.error(f"❌ Trade failed: {symbol}")
                    
            except Exception as e:
                logger.error(f"Error trading {symbol}: {str(e)}")
                failed_trades.append((symbol, f"Error: {str(e)}"))
                
            time.sleep(0.1) 
            
        # Print execution summary
        logger.info("\n=== Trade Summary ===")
        logger.info(f"Trades: {executed_trades}/{len(allocations)} successful")
        logger.info(f"Invested: {total_invested:.2f} USDT")
        logger.info(f"Remaining: {total_amount - total_invested:.2f} USDT")
        
        if failed_trades:
            logger.info("\nFailed Trades:")
            for symbol, reason in failed_trades:
                logger.info(f"  {symbol}: {reason}")
        
    except BinanceAPIException as e:
        logger.error(f"Binance API Error: {e.status_code} - {e.message}")
        raise
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise
    
if __name__ == "__main__":
    execute_trades()