import MetaTrader5 as mt5
import pandas as pd
from stable_baselines3 import PPO
import numpy as np
import pytz
from datetime import datetime
import time
import csv
import os
import telegram

model = PPO.load("ppo_rl_trading_agent")

TELEGRAM_TOKEN = "7554113052:AAFAhMr06qHW7SvmD9jDDklkjQxXPeoPdiI"
CHAT_ID = "2069893117"
bot = telegram.Bot(token=TELEGRAM_TOKEN)


SYMBOL = "XAUUSD"
LOT_SIZE = 0.1
MAX_TRADES_PER_DAY = 2
RISK_PERCENT = 10
ACCOUNT_BALANCE = 3581
SL_PIPS = 300
TP_MULTIPLIER = 3
LOG_FILE = "trade_log.csv"

if not mt5.initialize():
    raise Exception(f"MT5 init failed: {mt5.last_error()}")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "symbol", "bias", "action", "entry_price", "sl", "tp", "result"])


def get_ema(symbol, timeframe, period, n=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    close_prices = [r['close'] for r in rates]
    df = pd.DataFrame(close_prices, columns=["close"])
    df["ema"] = df["close"].ewm(span=period).mean()
    return df["ema"].iloc[-1]

def get_bias(symbol):
    ema_4h = get_ema(symbol, mt5.TIMEFRAME_H4, 50)
    ema_1h = get_ema(symbol, mt5.TIMEFRAME_H1, 50)
    ema_30m = get_ema(symbol, mt5.TIMEFRAME_M30, 50)
    current_price = mt5.symbol_info_tick(symbol).bid

    if current_price > ema_30m and ema_30m > ema_1h > ema_4h:
        return "Bullish"
    elif current_price < ema_30m and ema_30m < ema_1h < ema_4h:
        return "Bearish"
    return "Neutral"

def check_price_action(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)
    close = [r['close'] for r in rates]
    open_ = [r['open'] for r in rates]
    if close[-1] > open_[-1] and close[-2] < open_[-2]:
        return "Bullish Engulfing"
    elif close[-1] < open_[-1] and close[-2] > open_[-2]:
        return "Bearish Engulfing"
    return "No Signal"

def place_trade(symbol, bias):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if bias == "Bullish" else tick.bid
    point = mt5.symbol_info(symbol).point

    sl_points = 300 * point  # 30 pips
    tp_points = 900 * point  # 90 pips (1:3 RR)

    sl = price - sl_points if bias == "Bullish" else price + sl_points
    tp = price + tp_points if bias == "Bullish" else price - tp_points

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if bias == "Bullish" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "deviation": 20,
        "magic": 234000,
        "comment": "AI RL Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"{bias} trade placed: {price}")
        send_telegram(f"✅ Trade Placed: {bias} at {price:.2f} (SL: {sl:.2f}, TP: {tp:.2f})")
        log_trade(datetime.now(), symbol, bias, "buy" if bias == "Bullish" else "sell", price, sl, tp, "open")
        return True
    else:
        print(f"Trade failed: {result.comment}")
        return False


def log_trade(timestamp, symbol, bias, action, entry_price, sl, tp, result):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, symbol, bias, action, entry_price, sl, tp, result])

def build_live_observation(symbol, bias, action, entry_price, sl, tp):
    bias_val = 1 if bias == "Bullish" else -1 if bias == "Bearish" else 0
    action_val = 1 if action == "buy" else -1
    result_val = 0  # unknown yet in live
    obs = np.array([[bias_val, action_val, entry_price, sl, tp, result_val]], dtype=np.float32)
    return obs

def send_telegram(msg):
    bot.send_message(chat_id=CHAT_ID, text=msg)
         
def run_bot():
    trades_today = 0
    while trades_today < MAX_TRADES_PER_DAY:
        local_time = datetime.now(pytz.timezone('America/Nassau')).strftime("%H:%M")
        if "08:55" <= local_time <= "09:30":
            bias = get_bias(SYMBOL)
            print(f"Bias: {bias}")
            signal = check_price_action(SYMBOL)

            # Build live observation for AI
            tick = mt5.symbol_info_tick(SYMBOL)
            price = tick.ask if bias == "Bullish" else tick.bid
            point = mt5.symbol_info(SYMBOL).point
            sl = price - 500 * point if bias == "Bullish" else price + 500 * point
            tp = price + 1500 * point if bias == "Bullish" else price - 1500 * point

            live_obs = build_live_observation(
                SYMBOL,
                bias,
                "buy" if bias == "Bullish" else "sell",
                price,
                sl,
                tp
            )

            # AI decides: 0 = hold, 1 = buy, 2 = sell
            ai_action, _states = model.predict(live_obs, deterministic=False)

            # Approximate confidence (not perfect, but works)
            probs = model.policy.predict_values(live_obs)
            confidence = float(np.max(probs.detach().numpy()))

            send_telegram(f"🤖 AI says {bias} at {price:.2f} — Confidence: {confidence:.2f}")


            if ai_action == 1 and bias == "Bullish" and signal == "Bullish Engulfing":
                print("AI says BUY — conditions match. Executing trade...")
                if place_trade(SYMBOL, bias): trades_today += 1

            elif ai_action == 2 and bias == "Bearish" and signal == "Bearish Engulfing":
                print("AI says SELL — conditions match. Executing trade...")
                if place_trade(SYMBOL, bias): trades_today += 1

            else:
                print("AI suggests HOLD — no trade placed.")

        else:
            print(f"{local_time} - Waiting for 8:55–9:30 window...")
        time.sleep(60)
    send_telegram("📴 Max trades hit for the day. Bot is sleeping until tomorrow.")

        
run_bot()
