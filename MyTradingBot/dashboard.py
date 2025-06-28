import streamlit as st
import pandas as pd

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("📈 AI Trading Bot Dashboard")

df = pd.read_csv("trade_log.csv")

st.subheader("Latest Trades")
st.dataframe(df.tail(10))

st.subheader("Performance Metrics")
win_trades = df[df["result"] == "win"]
loss_trades = df[df["result"] == "loss"]
open_trades = df[df["result"] == "open"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Trades", len(df))
col2.metric("Wins", len(win_trades))
col3.metric("Losses", len(loss_trades))

st.line_chart(df["entry_price"])
