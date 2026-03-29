import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_data(ticker="CL=F", period="3y"):
    """Fetch price data and compute all indicators."""
    raw = yf.download(ticker, period=period, auto_adjust=True)
    if raw.empty:
        print("No data returned.")
        return None

    df = raw[["Close"]].copy()
    df.columns = ["price"]
    df.index.name = "date"
    df = df.dropna()
    df["daily_return"] = df["price"].pct_change()

    # SMA Crossover
    df["SMA50"]  = df["price"].rolling(50).mean()
    df["SMA200"] = df["price"].rolling(200).mean()

    # RSI (14-period)
    delta     = df["price"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period, 2 std devs)
    df["BB_mid"]   = df["price"].rolling(20).mean()
    df["BB_std"]   = df["price"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # MACD (12/26/9)
    ema12             = df["price"].ewm(span=12, adjust=False).mean()
    ema26             = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    print(f"Fetched {len(df)} trading days of data for {ticker}.")
    return df


# ── Core Backtesting Engine (with stop loss + portfolio tracking) ──────────────

def run_backtest(prices, signal_series, stop_loss_pct, starting_capital=1000.0):
    """
    Generic backtester with stop loss and dollar portfolio tracking.

    Logic:
    - Enter long when signal flips to 1
    - Set stop loss at entry_price * (1 - stop_loss_pct)
    - Exit if price hits stop loss OR signal flips to 0
    - Track portfolio value in dollars every day
    """
    portfolio   = starting_capital
    position    = 0
    entry_price = None
    stop_price  = None

    portfolio_values = []
    prices_arr       = prices.values
    signals_arr      = signal_series.reindex(prices.index).fillna(0).values

    for i in range(len(prices_arr)):
        price  = float(prices_arr[i])
        signal = int(signals_arr[i])

        if position == 1:
            # Stop loss triggered
            if price <= stop_price:
                trade_return = (stop_price - entry_price) / entry_price
                portfolio   *= (1 + trade_return)
                position     = 0
                entry_price  = None
                stop_price   = None
            # Signal says exit
            elif signal == 0:
                trade_return = (price - entry_price) / entry_price
                portfolio   *= (1 + trade_return)
                position     = 0
                entry_price  = None
                stop_price   = None

        # Enter new trade
        if position == 0 and signal == 1:
            position    = 1
            entry_price = price
            stop_price  = price * (1 - stop_loss_pct)

        # Mark-to-market portfolio value
        if position == 1:
            current_value = portfolio * (price / entry_price)
        else:
            current_value = portfolio

        portfolio_values.append(current_value)

    result = pd.DataFrame({"portfolio_value": portfolio_values}, index=prices.index)
    result["daily_return"] = result["portfolio_value"].pct_change()
    return result


# ── Signal Generators ──────────────────────────────────────────────────────────

def signal_sma(df):
    bt = df.dropna(subset=["SMA50", "SMA200"])
    return pd.Series(np.where(bt["SMA50"] > bt["SMA200"], 1, 0), index=bt.index)


def signal_rsi(df, oversold=30, overbought=70):
    bt = df.dropna(subset=["RSI"])
    position, signals = 0, []
    for rsi in bt["RSI"]:
        if rsi < oversold:
            position = 1
        elif rsi > overbought:
            position = 0
        signals.append(position)
    return pd.Series(signals, index=bt.index)


def signal_bollinger(df):
    bt = df.dropna(subset=["BB_upper", "BB_lower"])
    position, signals = 0, []
    for price, lower, upper in zip(bt["price"], bt["BB_lower"], bt["BB_upper"]):
        if price <= lower:
            position = 1
        elif price >= upper:
            position = 0
        signals.append(position)
    return pd.Series(signals, index=bt.index)


def signal_macd(df):
    bt = df.dropna(subset=["MACD", "MACD_signal"])
    return pd.Series(np.where(bt["MACD"] > bt["MACD_signal"], 1, 0), index=bt.index)


# ── Performance Metrics ────────────────────────────────────────────────────────

def compute_metrics(result, name, starting_capital):
    returns      = result["daily_return"].dropna()
    years        = len(result) / 252
    final_value  = result["portfolio_value"].iloc[-1]
    total_return = (final_value - starting_capital) / starting_capital
    ann_return   = (1 + total_return) ** (1 / years) - 1
    sharpe       = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    rolling_max  = result["portfolio_value"].cummax()
    max_drawdown = ((result["portfolio_value"] - rolling_max) / rolling_max).min()

    return {
        "name":            name,
        "final_value":     final_value,
        "total_return":    total_return,
        "ann_return":      ann_return,
        "sharpe":          sharpe,
        "max_drawdown":    max_drawdown,
        "portfolio_value": result["portfolio_value"],
        "index":           result.index,
    }


# ── Summary Table ──────────────────────────────────────────────────────────────

def print_summary(results, bh_result, starting_capital, ticker, stop_loss_pct):
    print(f"\n{'='*75}")
    print(f"  {ticker} | Capital: ${starting_capital:,.0f} | Stop Loss: {stop_loss_pct*100:.1f}%")
    print(f"{'='*75}")
    print(f"{'Strategy':<22} {'Final $':>10} {'Total Rtn':>10} {'Ann. Rtn':>10} {'Sharpe':>8} {'Max DD':>8}")
    print(f"{'-'*75}")
    for r in [bh_result] + results:
        print(f"{r['name']:<22} "
              f"${r['final_value']:>8,.0f} "
              f"{r['total_return']*100:>9.1f}% "
              f"{r['ann_return']*100:>9.1f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['max_drawdown']*100:>7.1f}%")
    print(f"{'='*75}")
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_return = max(results, key=lambda x: x["final_value"])
    print(f"\n  Best risk-adjusted : {best_sharpe['name']} (Sharpe: {best_sharpe['sharpe']:.2f})")
    print(f"  Best total return  : {best_return['name']} (${best_return['final_value']:,.0f})")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(df, results, bh_result, ticker, stop_loss_pct, starting_capital):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"{ticker} — Strategy Comparison  |  "
        f"Capital: ${starting_capital:,.0f}  |  Stop Loss: {stop_loss_pct*100:.1f}%",
        fontsize=14, fontweight="bold"
    )

    colors = {
        "Buy & Hold":      "#1f77b4",
        "SMA Crossover":   "#2ca02c",
        "RSI Reversion":   "#ff7f0e",
        "Bollinger Bands": "#9467bd",
        "MACD Crossover":  "#e377c2",
    }

    # ── Panel 1: Portfolio Value in Dollars ──
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[3, 1.5], hspace=0.4)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(bh_result["index"], bh_result["portfolio_value"],
             label=f"Buy & Hold (${bh_result['final_value']:,.0f})",
             color=colors["Buy & Hold"], linewidth=1.5, linestyle="--")
    for r in results:
        ax1.plot(r["index"], r["portfolio_value"],
                 label=f"{r['name']} (${r['final_value']:,.0f})",
                 color=colors[r["name"]], linewidth=1.3)
    ax1.axhline(starting_capital, color="gray", linewidth=0.8, linestyle=":",
                label=f"Starting capital (${starting_capital:,.0f})")
    ax1.set_title(f"Portfolio Value ($) — Starting from ${starting_capital:,.0f} with {stop_loss_pct*100:.1f}% Stop Loss")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax1.grid(alpha=0.3)

    # ── Panel 2: Bar Charts ──
    all_results  = [bh_result] + results
    names        = [r["name"] for r in all_results]
    bar_colors   = [colors.get(n, "#aec7e8") for n in names]
    final_values = [r["final_value"] for r in all_results]
    ann_returns  = [r["ann_return"] * 100 for r in all_results]
    sharpes      = [r["sharpe"] for r in all_results]

    ax5 = fig.add_subplot(gs[1, 0])
    bars = ax5.bar(names, final_values, color=bar_colors, edgecolor="white")
    ax5.axhline(starting_capital, color="gray", linewidth=0.8, linestyle="--")
    ax5.set_title("Final Portfolio Value ($)")
    ax5.set_ylabel("$")
    ax5.tick_params(axis="x", rotation=25, labelsize=7)
    for bar, val in zip(bars, final_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"${val:,.0f}", ha="center", va="bottom", fontsize=7)
    ax5.grid(axis="y", alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 1])
    bars = ax6.bar(names, ann_returns, color=bar_colors, edgecolor="white")
    ax6.axhline(0, color="gray", linewidth=0.7)
    ax6.set_title("Annualized Return (%)")
    ax6.set_ylabel("%")
    ax6.tick_params(axis="x", rotation=25, labelsize=7)
    for bar, val in zip(bars, ann_returns):
        offset = 0.2 if val >= 0 else -0.8
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax6.grid(axis="y", alpha=0.3)

    ax7 = fig.add_subplot(gs[1, 2])
    bars = ax7.bar(names, sharpes, color=bar_colors, edgecolor="white")
    ax7.axhline(0, color="gray", linewidth=0.7)
    ax7.set_title("Sharpe Ratio")
    ax7.set_ylabel("Sharpe")
    ax7.tick_params(axis="x", rotation=25, labelsize=7)
    for bar, val in zip(bars, sharpes):
        offset = 0.01 if val >= 0 else -0.05
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax7.grid(axis="y", alpha=0.3)

    fname = f"strategy_comparison_{ticker.replace('=', '')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {fname}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Strategy Backtester ===")
    ticker   = input("Ticker (e.g. CL=F, SPY, AAPL, NVDA) [default: CL=F]: ").strip() or "CL=F"
    period   = input("Time period (e.g. 1y, 2y, 3y, 5y) [default: 3y]: ").strip() or "3y"
    capital  = input("Starting capital in USD [default: 1000]: ").strip()
    capital  = float(capital) if capital else 1000.0
    sl_input = input("Stop loss % (e.g. 2 for 2%, between 1-5) [default: 2.5]: ").strip()
    sl_pct   = float(sl_input) / 100 if sl_input else 0.025

    print(f"\nRunning: {ticker} | {period} | ${capital:,.0f} | {sl_pct*100:.1f}% stop loss\n")

    df = fetch_data(ticker=ticker, period=period)
    if df is None:
        exit()

    # Generate signals
    sig_sma  = signal_sma(df)
    sig_rsi  = signal_rsi(df)
    sig_bb   = signal_bollinger(df)
    sig_macd = signal_macd(df)

    # Run backtests with stop loss
    prices   = df["price"]
    res_sma  = run_backtest(prices, sig_sma,  sl_pct, capital)
    res_rsi  = run_backtest(prices, sig_rsi,  sl_pct, capital)
    res_bb   = run_backtest(prices, sig_bb,   sl_pct, capital)
    res_macd = run_backtest(prices, sig_macd, sl_pct, capital)

    # Buy & hold benchmark
    bh_values = capital * (1 + df["daily_return"].fillna(0)).cumprod()
    years     = len(df) / 252
    bh_result = {
        "name":            "Buy & Hold",
        "final_value":     bh_values.iloc[-1],
        "total_return":    (bh_values.iloc[-1] - capital) / capital,
        "ann_return":      (bh_values.iloc[-1] / capital) ** (1 / years) - 1,
        "sharpe":          (df["daily_return"].mean() / df["daily_return"].std()) * np.sqrt(252),
        "max_drawdown":    ((bh_values - bh_values.cummax()) / bh_values.cummax()).min(),
        "portfolio_value": bh_values,
        "index":           bh_values.index,
    }

    results = [
        compute_metrics(res_sma,  "SMA Crossover",   capital),
        compute_metrics(res_rsi,  "RSI Reversion",   capital),
        compute_metrics(res_bb,   "Bollinger Bands", capital),
        compute_metrics(res_macd, "MACD Crossover",  capital),
    ]

    print_summary(results, bh_result, capital, ticker, sl_pct)
    plot_comparison(df, results, bh_result, ticker, sl_pct, capital)