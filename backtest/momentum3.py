import sys
from pathlib import Path
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import ta
from datetime import datetime

# Add the parent directory (AlgoTrade) to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

class MACDMomentum(Strategy): 

    # Strategy parameters
    trend_bars = 20  # Y bars for trend comparison
    adx_period = 14  # Period for ADX calculation
    adx_threshold = 20  # Minimum ADX value for trading
    volume_period = 100  # Period for volume average
    volume_threshold = 1.1  # Current volume should be 20% above average for shorts
    atr_period = 14  # Period for ATR calculation
    atr_ma_period = 100  # Period for ATR moving average
    atr_threshold = 1.1  # Current ATR should be above average
    momentum_threshold = 5.0  # Minimum MACD-Signal spread for momentum entries
    
    def init(self):
        # Calculate MACD indicators directly
        close = pd.Series(self.data.Close)
        
        # Calculate MACD using ta library
        self.macd_line = self.I(ta.trend.macd, close)
        self.macd_signal = self.I(ta.trend.macd_signal, close)
        
        # Calculate ADX components
        def calc_adx(high, low, close, period=14):
            # True Range
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Directional Movement
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            up_move[0] = 0
            down_move[0] = 0
            
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed TR and DM
            tr_smooth = pd.Series(tr).rolling(period).mean().values
            pos_dm_smooth = pd.Series(pos_dm).rolling(period).mean().values
            neg_dm_smooth = pd.Series(neg_dm).rolling(period).mean().values
            
            # Directional Indicators
            pdi = 100 * pos_dm_smooth / tr_smooth
            ndi = 100 * neg_dm_smooth / tr_smooth
            
            # ADX
            dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
            adx = pd.Series(dx).rolling(period).mean().values
            
            return adx
        
        # Calculate ADX
        self.adx = self.I(calc_adx, self.data.High, self.data.Low, self.data.Close, 
                         period=self.adx_period)
        
        # Initialize stop loss and take profit variables
        self.stop_loss = None
        self.take_profit = None
        
        # Risk parameters
        self.stop_loss_pct = 0.05  # 2% stop loss
        self.take_profit_pct = 0.15  # 4% take profit
        
        # Add volume moving average calculation
        self.volume_ma = self.I(lambda x: pd.Series(x).rolling(self.volume_period).mean(),
                              self.data.Volume)
        
        def calc_atr(high, low, close, period):
            # True Range calculation
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Calculate ATR using simple moving average
            atr = pd.Series(tr).rolling(period).mean().values
            return atr
        
        # Calculate ATR
        self.atr = self.I(calc_atr, self.data.High, self.data.Low, self.data.Close, 
                          period=self.atr_period)
        
        # Add ATR moving average calculation - now using self.atr instead of self.data.atr
        self.atr_ma = self.I(lambda x: pd.Series(x).rolling(self.atr_ma_period).mean(),
                             self.atr)
    
    def check_trend(self, direction='long'):
        """
        Check if the trend conditions are met
        direction: 'long' or 'short'
        Returns: True if trend conditions are met
        """
        current_close = self.data.Close[-1]
        y_bars_ago = self.data.Close[-self.trend_bars]
        half_y_bars_ago = self.data.Close[-self.trend_bars//2]
        
        if direction == 'long':
            return (current_close > y_bars_ago and 
                   current_close > half_y_bars_ago)
        else:  # short
            return (current_close < y_bars_ago and 
                   current_close < half_y_bars_ago)

    def check_volume(self, direction='long'):
        """
        Check if volume conditions are met
        direction: 'long' or 'short'
        Returns: True if volume conditions are met
        """
        current_volume = self.data.Volume[-1]
        avg_volume = self.volume_ma[-1]
        
        if direction == 'long':
            return True  # No volume filter for longs
        else:  # short
            return current_volume > (avg_volume * self.volume_threshold)

    def check_volatility(self):
        """
        Check if volatility conditions are met using ATR
        Returns: True if ATR is above its moving average
        """
        current_atr = self.atr[-1]
        avg_atr = self.atr_ma[-1]
        return current_atr > (avg_atr * self.atr_threshold)

    def next(self):
        # Get current position size
        pos_size = self.position.size if self.position else 0
        price = self.data.Close[-1]
        
        # Check stop loss and take profit
        if pos_size != 0:
            if self.stop_loss and (
                (pos_size > 0 and price <= self.stop_loss) or 
                (pos_size < 0 and price >= self.stop_loss)
            ):
                self.position.close()
                self.stop_loss = None
                self.take_profit = None
                return
                
            if self.take_profit and (
                (pos_size > 0 and price >= self.take_profit) or 
                (pos_size < 0 and price <= self.take_profit)
            ):
                self.position.close()
                self.stop_loss = None
                self.take_profit = None
                return

        # Get MACD and ADX values
        macd = self.macd_line[-1]
        signal = self.macd_signal[-1]
        prev_macd = self.macd_line[-2]
        prev_signal = self.macd_signal[-2]
        current_adx = self.adx[-1]
        
        # Check if ADX is strong enough
        if current_adx < self.adx_threshold:
            return
        
        # Long entry conditions
        if (macd > signal and prev_macd <= prev_signal and
            self.check_trend('long') and
            self.check_volatility()):
            
            if pos_size <= 0:
                if pos_size < 0:
                    self.position.close()
                self.buy()
                self.stop_loss = price * (1 - self.stop_loss_pct)
                self.take_profit = price * (1 + self.take_profit_pct)
                
        # Short entry conditions
        elif (macd < signal and prev_macd >= prev_signal and
              self.check_trend('short') and
              self.check_volume('short') and
              self.check_volatility()):
            
            if pos_size >= 0:
                if pos_size > 0:
                    self.position.close()
                self.sell()
                self.stop_loss = price * (1 + self.stop_loss_pct)
                self.take_profit = price * (1 - self.take_profit_pct)


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'ohlc' / '25_02_14_17_09.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Rename columns if needed
    column_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    data = data.rename(columns=column_mapping)

    bt = Backtest(data, MACDMomentum, cash=100000, commission=0.002)
    stats = bt.run()
    print(stats)

    trades_file = base_dir / 'trade_logs' / 'trades_macd.csv'
    trades_file.parent.mkdir(parents=True, exist_ok=True)
    stats['_trades'].to_csv(trades_file)
    print(f"Trade log saved to: {trades_file}")

    # bt.plot()