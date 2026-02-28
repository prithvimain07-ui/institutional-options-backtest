import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Institutional Backtest Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Institutional Options Backtest Dashboard")
st.markdown("Advanced Performance Analytics")

# ==========================
# CONFIG
# ==========================
INITIAL_CAPITAL = 1_000_000
LOT_SIZE = 50
LOTS = 1
IV_LOOKBACK = 50
IV_ENTRY = 0.35
IV_EXIT = 0.80
STOP_LOSS = 0.18
TAKE_PROFIT = 0.45
MAX_HOLD_BARS = 30
REALISTIC_COST_PCT = 0.0012


# ==========================
# COST FUNCTION
# ==========================
def trading_cost(price):
    turnover = price * LOT_SIZE * LOTS
    return turnover * REALISTIC_COST_PCT


# ==========================
# STRATEGY FUNCTION
# ==========================
def run_strategy(df):

    df.columns = df.columns.str.lower()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    atm = df[df["strike_offset"].astype(str).str.upper() == "ATM"]
    ce = atm[atm["type"].str.upper() == "CE"].reset_index(drop=True)
    pe = atm[atm["type"].str.upper() == "PE"].reset_index(drop=True)

    if len(ce) == 0 or len(pe) == 0:
        return 0, 0, 0, 0, [], []

    min_len = min(len(ce), len(pe))
    ce = ce.iloc[:min_len]
    pe = pe.iloc[:min_len]

    premium = ce["close"] + pe["close"]
    iv = ce["iv"]
    spot = ce["spot"]

    iv_pct = iv.rolling(IV_LOOKBACK).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    ema20 = spot.ewm(span=20).mean()
    ema50 = spot.ewm(span=50).mean()

    capital = INITIAL_CAPITAL
    equity_curve = []
    trade_pnls = []
    trade_dates = []

    trades = 0
    wins = 0
    total_cost = 0
    position = False
    bars_held = 0

    for i in range(IV_LOOKBACK + 50, len(premium)):

        if not position:

            spot_move = abs(spot.iloc[i] - spot.iloc[i-5]) / spot.iloc[i-5]
            trend_strength = abs(ema20.iloc[i] - ema50.iloc[i]) / spot.iloc[i]

            if (
                iv_pct.iloc[i] < IV_ENTRY
                and spot_move > 0.002
                and trend_strength > 0.001
            ):
                entry_price = premium.iloc[i]
                cost = trading_cost(entry_price)
                capital -= cost
                total_cost += cost

                stop = entry_price * (1 - STOP_LOSS)
                target = entry_price * (1 + TAKE_PROFIT)

                position = True
                bars_held = 0
                trades += 1

        else:
            current_price = premium.iloc[i]
            cost = trading_cost(current_price)
            bars_held += 1
            exit_trade = False

            if current_price >= target:
                wins += 1
                exit_trade = True
            elif current_price <= stop:
                exit_trade = True
            elif iv_pct.iloc[i] > IV_EXIT:
                if current_price > entry_price:
                    wins += 1
                exit_trade = True
            elif bars_held >= MAX_HOLD_BARS:
                if current_price > entry_price:
                    wins += 1
                exit_trade = True

            if exit_trade:
                pnl = (current_price - entry_price) * LOT_SIZE * LOTS
                capital += pnl - cost
                total_cost += cost
                trade_pnls.append(pnl)
                trade_dates.append(df["date"].iloc[i] if "date" in df.columns else i)
                position = False

        equity_curve.append(capital)

    net_pnl = capital - INITIAL_CAPITAL
    return net_pnl, trades, wins, total_cost, equity_curve, list(zip(trade_dates, trade_pnls))


# ==========================
# DATA CHECK
# ==========================
data_folder = "data"

if not os.path.exists(data_folder):
    st.error("❌ 'data' folder not found.")
    st.stop()

files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]

if len(files) == 0:
    st.warning("⚠️ No CSV files found.")
    st.stop()


# ==========================
# RUN BUTTON
# ==========================
if st.button("🚀 Run Backtest"):

    with st.spinner("Running Strategy..."):

        total_pnl = 0
        total_trades = 0
        total_wins = 0
        total_costs = 0
        combined_equity = []
        all_trades = []

        for file in files:
            df = pd.read_csv(file)
            pnl, trades, wins, cost, equity_curve, trade_data = run_strategy(df)

            total_pnl += pnl
            total_trades += trades
            total_wins += wins
            total_costs += cost
            combined_equity.extend(equity_curve)
            all_trades.extend(trade_data)

        final_capital = INITIAL_CAPITAL + total_pnl
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # ==========================
        # MAX DRAWDOWN
        # ==========================
        equity_series = pd.Series(combined_equity)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        # ==========================
        # SHARPE RATIO
        # ==========================
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

        # ==========================
        # MONTHLY RETURNS
        # ==========================
        if all_trades:
            trades_df = pd.DataFrame(all_trades, columns=["Date", "PnL"])
            trades_df["Date"] = pd.to_datetime(trades_df["Date"], errors="coerce")
            trades_df = trades_df.dropna()
            trades_df["Month"] = trades_df["Date"].dt.to_period("M")
            monthly_returns = trades_df.groupby("Month")["PnL"].sum()
        else:
            monthly_returns = pd.Series()

    st.success("Backtest Completed ✅")

    # ==========================
    # METRICS
    # ==========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Net PnL", f"₹ {round(total_pnl,2)}")
    col2.metric("Max Drawdown", f"{round(max_dd,2)}%")
    col3.metric("Sharpe Ratio", round(sharpe,2))
    col4.metric("Win Rate", f"{round(win_rate,2)}%")

    # ==========================
    # EQUITY CURVE
    # ==========================
    st.subheader("📈 Equity Curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=combined_equity,
        mode='lines',
        line=dict(color='green' if total_pnl >= 0 else 'red', width=3),
        name="Equity"
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # SCATTER TRADE PLOT
    # ==========================
    if all_trades:
        st.subheader("📍 Trade PnL Scatter")

        trades_df = pd.DataFrame(all_trades, columns=["Date", "PnL"])

        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=trades_df["PnL"],
            mode='markers',
            marker=dict(
                color=np.where(trades_df["PnL"] > 0, 'green', 'red'),
                size=8
            )
        ))

        scatter_fig.update_layout(height=400)
        st.plotly_chart(scatter_fig, use_container_width=True)

    # ==========================
    # MONTHLY RETURNS TABLE
    # ==========================
    st.subheader("📆 Monthly Returns")

    if not monthly_returns.empty:
        st.dataframe(monthly_returns)
    else:
        st.write("No monthly return data available.")

    st.subheader("💰 Total Trading Costs")
    st.write(f"₹ {round(total_costs,2)}")