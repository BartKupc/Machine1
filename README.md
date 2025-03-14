# Machine1 - Cryptocurrency Market Analysis Engine

A powerful cryptocurrency market analysis engine that uses Principal Component Analysis (PCA) to identify market regimes and generate trading signals in real-time.

## Overview

Machine1 is the core analysis component that:
- Connects to cryptocurrency exchanges (primarily Bitget) to retrieve market data
- Calculates technical indicators and other market features
- Performs PCA to reduce dimensionality and extract meaningful market patterns
- Classifies market regimes based on principal components
- Generates trading signals based on the identified market conditions

## Features

- **Real-time Market Classification**: Categorizes markets into regimes like "Strong Trending Market", "Momentum Without Trend", and "Choppy/Noisy Market"
- **Principal Component Analysis**: Reduces complex market data into interpretable components:
  - PC1: Captures momentum strength (MACD & RSI)
  - PC2: Captures trend strength (ADX & volatility)
  - PC3: Identifies divergences and mean reversion opportunities
- **Trading Signals**: Generates Long, Short, Hold, or Avoid Trading signals
- **API Integration**: Connects with cryptocurrency exchanges through CCXT

## Installation

### Prerequisites
- Python 3.8+
- pip
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Machine1.git
cd Machine1
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a configuration file:
```bash
# Copy the example config
cp example_config.json config/config.json
# Edit the config file with your exchange API keys
```

## Configuration

### API Keys Setup

Create a `config.json` file in the `config` directory with the following structure:

```json
{
  "bitget": {
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET_KEY",
    "password": "YOUR_API_PASSWORD",
    "options": {
      "defaultType": "future"
    }
  },
  "other_settings": {
    "update_interval_hours": 1,
    "symbol": "ETH/USDT:USDT",
    "timeframe": "1h",
    "candles_back": 300
  }
}
```

## Usage

To run the market interpreter directly:

```bash
python strategy/market_interpreter.py
```

This will output the current market condition, including the market regime, trading signal, and principal component values.

## How It Works

### Market Classification Logic

The system classifies market conditions based on the values of PC1 and PC2:

- **Strong Trending Market**: PC1 > PC1_mean AND PC2 > PC2_mean
- **Momentum Without Trend**: PC1 > PC1_mean AND PC2 < PC2_mean
- **Choppy/Noisy Market**: PC1 < PC1_mean AND PC2 < PC2_mean
- **Undefined**: PC1 < PC1_mean AND PC2 > PC2_mean

### PC3 Interpretation

PC3 captures divergences between momentum and oscillator indicators:

- **Positive PC3**: High MACD, Low RSI → Strong momentum but overbought (potential correction)
- **Negative PC3**: Low MACD, High RSI → Weak momentum but oversold (potential reversal)

### Trading Strategy

The system provides trading signals based on these components:
- Go **Long** when PC1 is positive and PC2 is rising
- Go **Short** when PC1 is negative and PC2 is dropping
- **Hold** in transitional phases
- **Avoid Trading** when PC3 indicates high risk of mean reversion

## AWS Deployment

For AWS deployment instructions, see the AWS deployment section in the documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the sklearn and pandas teams for their excellent libraries
- Bitget exchange for providing the trading API 