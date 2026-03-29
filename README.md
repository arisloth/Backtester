# Backtester — Multi-Strategy Trading Analysis

A Python-based backtesting framework that compares four trading strategies on any ticker using real market data. Built to analyze how different strategies perform across asset classes: commodities vs equities.

## Strategies

| Strategy | Logic |
|---|---|
| SMA Crossover | Long when SMA50 > SMA200, exit when it crosses below |
| RSI Mean Reversion | Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought) |
| Bollinger Bands | Buy at lower band, sell at upper band |
| MACD Crossover | Long when MACD line crosses above signal line |

## Features

- Configurable starting capital, stop loss percentage, ticker, and time period
- Fixed stop loss per trade to model realistic risk management
- Portfolio value tracked in dollars across the full period
- Performance metrics: annualized return, Sharpe ratio, max drawdown
- Benchmarked against buy & hold
- Multi-panel chart output saved as PNG

## Key Finding

RSI mean reversion outperformed on WTI Crude Oil (CL=F) while Bollinger Bands performed best on S&P 500 (SPY), showing that strategy selection should depend on the underlying asset's characteristics and not applied blindly.

## Usage

```bash
pip install yfinance pandas numpy matplotlib
python stock_strat_comparison.py
```

You'll be prompted to enter:
- Ticker (e.g. `CL=F`, `SPY`, `AAPL`, `NVDA`)
- Time period (e.g. `1y`, `2y`, `3y`, `5y`)
- Starting capital in USD
- Stop loss percentage (1-5%)

## Example Output

```
=======================================================================
  SPY | Capital: $1,000 | Stop Loss: 2.5%
=======================================================================
Strategy               Final $   Total Rtn   Ann. Rtn   Sharpe   Max DD
-----------------------------------------------------------------------
Buy & Hold            $1,666      66.6%      18.6%      1.21    -15.2%
Bollinger Bands       $1,505      50.5%      14.7%      1.17     -6.9%
MACD Crossover        $1,220      22.0%       6.9%      0.92    -10.1%
RSI Reversion         $1,259      25.9%       8.0%      0.82    -17.3%
SMA Crossover           $920      -8.0%      -2.8%     -0.08     -8.2%
=======================================================================
```

## Built With

- Python, Pandas, NumPy, Matplotlib, yfinance

## Author

[Arian Popal](https://github.com/arisloth)