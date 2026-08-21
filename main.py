import pandas as pd
import numpy as np
import cufflinks as cf
import yfinance as yf
#from pandas_datareader import data
from datetime import datetime
import datetime
#import matplotlib.pyplot as plt
#from pandas.tseries.frequencies import to_offset
#import mplfinance as mpf
#from mpl_finance import candlestick_ohlc
import yfinance
#import matplotlib.dates as mpl_dates
##from fbprophet import Prophet
import streamlit as st
from plotly import graph_objs as go
from plotly import express as px
##from fbprophet.plot import plot_plotly
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting import Backtest

import codecs

import streamlit.components.v1 as stc

# import prophet
# from prophet.plot import plot_plotly



#####################################################################################

#start = "2020-06-01"
#end  = datetime.today()

#ema_10 = 10
#ema_20 = 20
#ema_50 = 50
#ema_100 = 100
ema_10 = 10
ema_20 = 20

#####################################################################################

st.title('Stock App')

user_input = st.text_input('Enter Ticker','aapl')
user_input_start = st.date_input('Enter Start Date',datetime.date(2019, 1, 1))
user_input_end = st.date_input('Enter End Date',datetime.date(2026, 8, 1))

#####################################################################################

user_ticker = user_input.strip().upper()
if not user_ticker:
    st.warning("Please enter a ticker symbol, for example AAPL or 1113.HK.")
    st.stop()

try:
    df = yf.download(
        user_ticker,
        start=user_input_start,
        end=user_input_end,
        progress=False,
    )
except Exception:
    st.warning(
        f"Unable to download market data for {user_ticker}. "
        "Please check the ticker symbol and try again."
    )
    st.stop()

if df is None or df.empty:
    st.warning(
        f"No market data was found for {user_ticker}. "
        "Please check the ticker symbol, such as 1113.HK instead of 1113."
    )
    st.stop()

# yfinance 1.3.0 returns MultiIndex columns for a single ticker by default.
# Flatten them once so Plotly, cufflinks, and the backtesting code receive
# ordinary string column names.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel("Ticker")

required_columns = {"Open", "High", "Low", "Close"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.warning(
        f"Market data for {user_ticker} is missing required price columns. "
        "Please try another ticker."
    )
    st.stop()

df = df.reset_index().dropna()
df1 = df.copy()
tickerData = yf.Ticker(user_ticker) # Get ticker data

try:
    ticker_info = tickerData.info
except Exception:
    ticker_info = {}

company_name = (
    ticker_info.get("longName")
    or ticker_info.get("shortName")
    or user_ticker
)
company_country = ticker_info.get("country") or "Information unavailable"
st.subheader(company_name)
st.caption(f"Ticker: {user_ticker}  |  Country: {company_country}")

weekly = (
    df1.set_index("Date")
       .resample("W-Fri")
       .agg({
           "Open": "first",
           "High": "max",
           "Low": "min",
           "Close": "last"
       })
)

weekly1 = weekly                                 
weekly = weekly.reset_index()

df2 = df                                 

#####################################################################################

#CHANGE A LOT AAAAAAAAA, PUT IMPORTANT PARAMETER BEFORE THIS

try:
    hist = tickerData.history(start=user_input_start,end=user_input_end)
except Exception:
    hist = pd.DataFrame()

if "Dividends" in hist.columns and hist["Dividends"].notna().any():
    hist1 = hist[["Dividends"]].copy()
else:
    hist1 = pd.DataFrame()
#hist2 = hist[["Dividends","Stock Splits"]]

#####################################################################################

#string_logo = '<img src=%s>' % tickerData.info['logo_url']
#st.markdown(string_logo, unsafe_allow_html=True)

#string_name = tickerData.info['longName']
#st.header(string_name)

#string_summary = tickerData.info['longBusinessSummary']
#st.info(string_summary)

#####################################################################################

def display_table(data, rows=None):
    table = data.tail(rows).copy() if rows else data.copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(4)
    st.dataframe(table, width="stretch", hide_index=True)


st.subheader('Market Data')
display_table(df[["Date", "Open", "High", "Low", "Close", "Volume"]], rows=30)

df["ema_10"] = df.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
df["ema_20"] = df.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
df=df.dropna()

#####################################################################################

#DAILY CHART

st.subheader(' ')
st.subheader('Day Chart')
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
	st.plotly_chart(fig, key="daily_line_chart")
	
plot_raw_data()

#####################################################################################

#Dividends and stock splits

if not hist1.empty:
    fig3 = px.line(hist1, title="Dividends")
    st.plotly_chart(fig3, key="dividends_chart")
else:
    st.info(f"No dividend data is available for {user_input.upper()} in the selected period.")

#####################################################################################

# Candlestick chart
def plot_candlestick(data, title):
    chart = data.copy().dropna(subset=["Open", "High", "Low", "Close"])
    x = chart["Date"] if "Date" in chart.columns else chart.index

    chart["ema_10"] = chart["Close"].ewm(span=ema_10, min_periods=ema_10).mean()
    chart["ema_20"] = chart["Close"].ewm(span=ema_20, min_periods=ema_20).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x,
        open=chart["Open"],
        high=chart["High"],
        low=chart["Low"],
        close=chart["Close"],
        name="Candle"
    ))
    fig.add_trace(go.Scatter(x=x, y=chart["ema_10"], name="EMA 10", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x, y=chart["ema_20"], name="EMA 20", line=dict(color="red")))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig


fig4 = plot_candlestick(df, user_input)
st.plotly_chart(fig4, key="daily_candlestick_chart")

#####################################################################################

st.subheader('Week Chart')
display_table(weekly, rows=20)

#WEEKLY CHART

st.subheader(' ')
fig5 = plot_candlestick(weekly, user_input)
st.plotly_chart(fig5, key="weekly_candlestick_chart")

#####################################################################################

# #Forecast

# st.subheader('Forecast')
# df[['ds','y']]=df[['Date','Adj Close']]
# model=Prophet()
# model.fit(df)
# future = model.make_future_dataframe(periods=365)
# forecast = model.predict(future)

# fig6=plot_plotly(model,forecast)
# st.plotly_chart(fig6)

#####################################################################################

#Recommendations

#st.subheader('Recommendations')
#st.write(tickerData.recommendations.tail(30))

#####################################################################################

#Strategy WEEK & DAY

weekly["ema_10"] = weekly.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
weekly["ema_20"] = weekly.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
weekly["position"] = np.where(weekly["ema_10"] > weekly["ema_20"], 1, -1 )

df["ema_10"] = df.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
df["ema_20"] = df.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
df["position"] = np.where(df["ema_10"] > df["ema_20"], 1, -1 )


weekly_position = weekly[["Date","Close","ema_10","ema_20","position"]]
daily_position = df[["Date","Close","ema_10","ema_20","position"]]

st.subheader('Position')    

display_table(weekly_position, rows=26)
st.subheader('')
st.subheader('')


display_table(daily_position, rows=10)

#####################################################################################

st.subheader('')
st.subheader('')

def openHTML(test_html,width=900,height=800):
    file=codecs.open(test_html,'r')
    page = file.read()
    stc.html(page,width=width,height=height)

#BackTest

class SmaCross(SignalStrategy, TrailingStrategy):
    s1 = 10
    s2 = 20
    
    def init(self):
        super().init()
        
        sma1 = self.I(SMA, self.data.Close, self.s1)
        sma2 = self.I(SMA, self.data.Close, self.s2)

        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only

        entry_size = signal * .95

        self.set_signal(entry_size = entry_size)
        
        self.set_atr_periods(20)
        self.set_trailing_sl(2) #ATR stop loss (20*2)

      




#####################################################################################

daily_backtest_data = df1.set_index("Date").copy()
daily_backtest_data.index = pd.to_datetime(daily_backtest_data.index)
bt_daily = Backtest(daily_backtest_data,SmaCross,cash=1000, commission=0.002,trade_on_close=True)
bt_daily_stats = bt_daily.run()
print("Daily backtest results:")
print(bt_daily_stats.to_string())

def display_backtest_stats(stats, title):
    summary = stats.drop(labels=["_equity_curve", "_trades"], errors="ignore")
    summary = summary.rename_axis("Metric").reset_index(name="Value")

    def format_stat_value(metric, value):
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "—"
        if metric in {"Start", "End"}:
            return str(value).split(" ")[0]
        if "[%]" in metric:
            return f"{float(value):.2f}%"
        if "[$]" in metric:
            return f"${float(value):,.2f}"
        if metric == "# Trades":
            return f"{int(value):,}"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.2f}"
        return str(value)

    summary["Value"] = [
        format_stat_value(metric, value)
        for metric, value in zip(summary["Metric"], summary["Value"])
    ]
    st.subheader(title)

    key_metrics = [
        ("Start", "Start Date"),
        ("End", "End Date"),
        ("Return [%]", "Return"),
        ("Buy & Hold Return [%]", "Buy & Hold"),
        ("Equity Final [$]", "Final Equity"),
        ("Max. Drawdown [%]", "Max Drawdown"),
        ("Sharpe Ratio", "Sharpe"),
        ("# Trades", "Trades"),
        ("Win Rate [%]", "Win Rate"),
    ]
    metric_columns = st.columns(3)
    for position, (metric_name, display_name) in enumerate(key_metrics):
        value = summary.loc[summary["Metric"] == metric_name, "Value"]
        if not value.empty:
            display_value = value.iloc[0]
            with metric_columns[position % 3]:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid rgba(128,128,128,.35);
                        border-radius: 10px;
                        padding: 10px 14px;
                        margin: 4px 0 10px 0;
                        min-height: 66px;
                    ">
                        <div style="font-size: 13px; opacity: .72;">{display_name}</div>
                        <div style="font-size: 22px; font-weight: 600; white-space: nowrap;">
                            {display_value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("All backtest metrics"):
        st.dataframe(summary, width="stretch", hide_index=True)


display_backtest_stats(bt_daily_stats, 'Daily Backtest Results')

#btINFO=bt.run()
#st.write(btINFO)    

bt_daily.plot(filename="SmaCross.html", open_browser=False)
st.subheader('Daily Chart BT')  
openHTML('SmaCross.html')

#####################################################################################

class SmaCross1(SignalStrategy, TrailingStrategy):
    s1 = 10
    s2 = 20
    
    def init(self):
        super().init()
        
        sma1 = self.I(SMA, self.data.Close, self.s1)
        sma2 = self.I(SMA, self.data.Close, self.s2)

        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only

        entry_size = signal * .95

        self.set_signal(entry_size = entry_size)
        
        self.set_atr_periods(20)
        self.set_trailing_sl(2) #ATR stop loss (20*2)



#####################################################################################

bt_weekly = Backtest(weekly1, SmaCross1,cash=1000, commission=0.002,trade_on_close=True)
bt_weekly_stats = bt_weekly.run()
print("Weekly backtest results:")
print(bt_weekly_stats.to_string())
display_backtest_stats(bt_weekly_stats, 'Weekly Backtest Results')


bt_weekly.plot(filename="SmaCross1.html", open_browser=False)






st.subheader('Weekly Chart BT')  
openHTML('SmaCross1.html')
