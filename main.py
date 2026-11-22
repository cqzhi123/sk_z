import pandas as pd
import numpy as np
# import cufflinks as cf
import yfinance as yf
#from pandas_datareader import data
from datetime import datetime
from zoneinfo import ZoneInfo
import datetime
#import matplotlib.pyplot as plt
#from pandas.tseries.frequencies import to_offset
#import mplfinance as mpf
#from mpl_finance import candlestick_ohlc
import yfinance
from html import escape
from urllib.parse import quote, urlparse
#import matplotlib.dates as mpl_dates
##from fbprophet import Prophet
import streamlit as st
from plotly import graph_objs as go
from plotly import express as px
##from fbprophet.plot import plot_plotly
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.lib import FractionalBacktest, SignalStrategy, TrailingStrategy

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


def EMA(values, period):
    """Return an exponential moving average for backtesting.py indicators."""
    return (
        pd.Series(values)
        .ewm(span=period, min_periods=period, adjust=False)
        .mean()
        .to_numpy()
    )

#####################################################################################

st.markdown("<div style='height: 7.3rem;'></div>", unsafe_allow_html=True)
st.title('Stock App')

ticker_options = [
    "GC=F", "CL=F", "BTC-USD", "VOO", "^HSI", "^KLSE",
]
ticker_from_url = st.query_params.get("ticker", "")
start_date_from_url = st.query_params.get("start", "")
end_date_from_url = st.query_params.get("end", "")


def get_date_from_url(value, fallback):
    try:
        return datetime.date.fromisoformat(str(value)) if value else fallback
    except ValueError:
        return fallback


start_date_default = get_date_from_url(
    start_date_from_url, datetime.date(2019, 1, 1)
)
today_malaysia = datetime.datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).date()
if (
    ticker_from_url
    and st.session_state.get("_ticker_from_url") != ticker_from_url
):
    st.session_state.ticker_selector = ticker_from_url
    st.session_state._ticker_from_url = ticker_from_url

if "ticker_selector" not in st.session_state:
    st.session_state.ticker_selector = "1155.KL"
if (
    start_date_from_url
    and st.session_state.get("_start_date_from_url") != start_date_from_url
):
    st.session_state.start_date_selector = start_date_default
    st.session_state._start_date_from_url = start_date_from_url
if "start_date_selector" not in st.session_state:
    st.session_state.start_date_selector = start_date_default
# Default to today; retain a user's selected historical end date.
end_date_min = datetime.date(1900, 1, 1)
end_date_default = max(
    end_date_min,
    min(get_date_from_url(end_date_from_url, today_malaysia), today_malaysia),
)
if st.session_state.get("_end_date_from_url") != end_date_from_url:
    st.session_state.end_date_selector = end_date_default
    st.session_state._end_date_from_url = end_date_from_url
elif "end_date_selector" not in st.session_state:
    st.session_state.end_date_selector = end_date_default
elif (
    not end_date_from_url
    and st.session_state.end_date_selector == st.session_state.get("_end_date_today")
):
    # Roll the default forward if the app remains open across midnight.
    st.session_state.end_date_selector = today_malaysia
st.session_state._end_date_today = today_malaysia

user_input = st.selectbox(
    "Enter Ticker",
    options=ticker_options,
    index=None,
    accept_new_options=True,
    help="Choose a listed ticker or type any Yahoo Finance symbol.",
    key="ticker_selector",
)
st.caption(
    "Sample tickers — "
    "US: AAPL, GOOGL, MSFT  |  "
    "Malaysia: 1155.KL, 5183.KL, 5296.KL  |  "
    "Hong Kong: 0700.HK, 1810.HK, 9988.HK"
)
user_input_start = st.date_input('Enter Start Date', key="start_date_selector")
user_input_end = st.date_input(
    'Enter End Date',
    key="end_date_selector",
    min_value=end_date_min,
    max_value=today_malaysia,
    help="Defaults to today in Malaysia. You can select an earlier date.",
)
# Yahoo Finance's end date is exclusive; include the selected end date.
download_end_date = user_input_end + datetime.timedelta(days=1)
if user_input_start > user_input_end:
    st.warning("Start Date cannot be later than End Date.")
    st.stop()
date_range_label = (
    f"{user_input_start.strftime('%Y-%m-%d')} to "
    f"{user_input_end.strftime('%Y-%m-%d')}"
)

NAVIGATION_VIEWS = ("Overview", "Daily", "Weekly", "Monthly", "Summary", "Fundamental", "Dividend", "News")
backtest_view = st.query_params.get("view", "Overview")
if backtest_view not in NAVIGATION_VIEWS:
    backtest_view = "Overview"

#####################################################################################

user_ticker = (user_input or "").strip().upper()
if not user_ticker:
    st.info("Choose or type a ticker symbol to begin, for example aapl or 1113.hk or 1155.kl.")
    st.stop()

try:
    df = yf.download(
        user_ticker,
        start=user_input_start,
        end=download_end_date,
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

if "Volume" not in df.columns:
    df["Volume"] = 0
if "Volume" in df.columns:
    df["Volume"] = df["Volume"].fillna(0)
df = df.reset_index().dropna(subset=["Open", "High", "Low", "Close"])
if df.empty:
    st.warning("No valid price bars were returned for the selected range.")
    st.stop()
df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
df = df.sort_values("Date").reset_index(drop=True)
df1 = df.copy()
tickerData = yf.Ticker(user_ticker) # Get ticker data

import logging

logger = logging.getLogger(__name__)

# Backup company information when Yahoo Finance is unavailable.
COMPANY_FALLBACK = {
    "1155.KL": {
        "longName": "Malayan Banking Berhad (Maybank)",
        "country": "Malaysia",
        "website": "https://www.maybank.com",
    },
}


@st.cache_data(ttl=900, show_spinner=False)
def get_company_info(symbol):
    symbol = symbol.strip().upper()
    info = {}
    fallback = COMPANY_FALLBACK.get(symbol, {})

    # Method 1: Yahoo Finance company information.
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        logger.warning("Ticker.info failed for %s: %r", symbol, exc)

    # Method 2: Yahoo Finance Search for missing company names.
    if not (info.get("longName") or info.get("shortName") or fallback.get("longName")):
        try:
            results = yf.Search(symbol, max_results=5).quotes or []
            for result in results:
                if str(result.get("symbol", "")).upper() == symbol:
                    name = result.get("longname") or result.get("shortname")
                    if name:
                        info["longName"] = name
                    break
        except Exception as exc:
            logger.warning("Ticker search failed for %s: %r", symbol, exc)

    # Method 3: Merge backup values and available Yahoo Finance values.
    return {**fallback, **{key: value for key, value in info.items() if value is not None and value != ""}}


ticker_info = get_company_info(user_ticker)

company_name = (
    ticker_info.get("longName")
    or ticker_info.get("shortName")
    or user_ticker
)

# A listing market is not necessarily the company's country of domicile.
market = {
    ".KL": "Malaysia (listing)",
    ".HK": "Hong Kong (listing)",
    ".SI": "Singapore (listing)",
}
company_country = ticker_info.get("country") or next(
    (name for suffix, name in market.items() if user_ticker.endswith(suffix)),
    "Information unavailable",
)

company_website = ticker_info.get("website") or ""
company_logo_url = ticker_info.get("logo_url")

# Yahoo Finance may not provide a logo URL; use the company's website favicon.
if not company_logo_url and company_website:
    website = str(company_website).strip()
    if "://" not in website:
        website = "https://" + website
    company_domain = urlparse(website).hostname
    if company_domain:
        company_logo_url = (
            "https://www.google.com/s2/favicons?domain="
            f"{quote(company_domain, safe='')}&sz=128"
        )

company_logo_html = '<span class="stock-company-logo-fallback" aria-hidden="true">↗</span>'
if company_logo_url:
    company_logo_html = (
        f'<img class="stock-company-logo" '
        f'src="{escape(str(company_logo_url), quote=True)}" '
        f'alt="{escape(str(company_name), quote=True)} logo" '
        'onerror="this.remove()">'
    )
display_company_name = escape(str(company_name))
display_company_country = escape(str(company_country))
display_ticker = escape(str(user_ticker))
ticker_url_value = quote(user_ticker, safe="")
start_date_url_value = user_input_start.isoformat()
# Keep custom dates when navigating, but leave the default date dynamic.
end_date_url_param = (
    f"&amp;end={user_input_end.isoformat()}"
    if user_input_end != today_malaysia else ""
)
navigation_html = "".join(
    '<a class="stock-view-button {active}" '
    'href="?view={view}&amp;ticker={ticker}&amp;start={start}{end}" '
    'target="_self" {aria}>{view}</a>'.format(
        active="active" if backtest_view == view else "",
        view=view, ticker=ticker_url_value, start=start_date_url_value,
        end=end_date_url_param, aria='aria-current="page"' if backtest_view == view else "",
    )
    for view in NAVIGATION_VIEWS
)
st.markdown(
    f"""
    <style>
    .stock-company-header {{
        position: fixed;
        top: 3.25rem;
        left: 0;
        z-index: 999999;
        box-sizing: border-box;
        width: 100%;
        padding: 0.6rem max(1rem, calc((100vw - 76rem) / 2));
        border-bottom: 1px solid rgba(128, 128, 128, 0.35);
        background: rgba(14, 17, 23, 0.96);
        color: #fafafa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
    }}
    .stock-company-header-inner {{
        display: flex;
        align-items: center;
        flex-direction: column;
        align-items: stretch;
        gap: 0.55rem;
    }}
    .stock-company-details {{
        min-width: 0;
    }}
    .stock-company-identity {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        min-width: 0;
    }}
    .stock-company-logo {{
        width: 42px;
        height: 42px;
        flex-shrink: 0;
        border-radius: 8px;
        background: #ffffff;
        object-fit: contain;
        padding: 3px;
    }}
    .stock-company-logo-fallback {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        flex-shrink: 0;
        border: 1px solid rgba(250, 250, 250, 0.35);
        border-radius: 8px;
        background: #1f2937;
        color: #60a5fa;
        font-size: 1.45rem;
        font-weight: 700;
    }}
    .stock-company-name {{
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .stock-company-meta {{
        margin-top: 0.2rem;
        font-size: 0.82rem;
        opacity: 0.75;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .stock-company-header-spacer {{
        height: 0;
    }}
    .stock-view-switcher {{
        display: flex;
        gap: 0.4rem;
        width: 100%;
        min-width: 0;
        overflow-x: auto;
        padding-bottom: 0.2rem;
        scrollbar-width: thin;
    }}
    .stock-view-button {{
        display: inline-block;
        flex: 0 0 auto;
        white-space: nowrap;
        padding: 0.42rem 0.8rem;
        border: 1px solid rgba(250, 250, 250, 0.35);
        border-radius: 0.45rem;
        color: #fafafa !important;
        font-size: 0.86rem;
        font-weight: 600;
        text-decoration: none !important;
    }}
    .stock-view-button.active {{
        border-color: #ff4b4b;
        background: #ff4b4b;
    }}
    @media (max-width: 640px) {{
        .stock-company-header {{ padding: 0.5rem 0.7rem; }}
        .stock-company-name {{ font-size: 1rem; }}
        .stock-company-meta {{ font-size: 0.72rem; }}
        .stock-view-button {{ padding: 0.35rem 0.65rem; }}
    }}
    </style>
    <div class="stock-company-header">
        <div class="stock-company-header-inner">
            <div class="stock-company-identity">
                {company_logo_html}
                <div class="stock-company-details">
                    <div class="stock-company-name">{display_company_name}</div>
                    <div class="stock-company-meta">
                        {display_ticker} · {display_company_country} · {date_range_label}
                    </div>
                </div>
            </div>
            <div class="stock-view-switcher">
                {navigation_html}
            </div>
        </div>
    </div>
    <div class="stock-company-header-spacer"></div>
    """,
    unsafe_allow_html=True,
)


weekly_aggregations = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
}
if "Volume" in df1.columns:
    weekly_aggregations["Volume"] = "sum"

weekly = (
    df1.set_index("Date")
       .resample("W-Fri")
       .agg(weekly_aggregations)
       .dropna(subset=["Open", "High", "Low", "Close"])
)

if "Volume" not in weekly.columns:
    weekly["Volume"] = 0

weekly1 = weekly                                 
weekly = weekly.reset_index()

df2 = df                                 

#####################################################################################

#CHANGE A LOT AAAAAAAAA, PUT IMPORTANT PARAMETER BEFORE THIS

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


if backtest_view == "Daily":
    st.markdown(f"### Daily Market Data ({date_range_label})")
    display_table(df[["Date", "Open", "High", "Low", "Close", "Volume"]], rows=10)

# st.subheader('Weekly Market Data')
# display_table(weekly[["Date", "Open", "High", "Low", "Close"]], rows=30)

df["ema_10"] = df.Close.ewm(span = ema_10, min_periods = ema_10).mean() 
df["ema_20"] = df.Close.ewm(span = ema_20, min_periods = ema_20).mean() 
# Keep the original price bars; EMA warm-up values may be NaN.

#####################################################################################

#DAILY CHART

# st.subheader(' ')
# st.subheader('Day Chart')
# def plot_raw_data():
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
# 	st.plotly_chart(fig, key="daily_line_chart")
	
# plot_raw_data()

#####################################################################################

#Dividends and stock splits

# if not hist1.empty:
#     fig3 = px.line(hist1, title="Dividends")
#     st.plotly_chart(fig3, key="dividends_chart")
# else:
#     st.info(f"No dividend data is available for {user_input.upper()} in the selected period.")

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

# New navigation pages. Prices use the selected history; company fundamentals
# and news are current Yahoo Finance snapshots, not historical point-in-time data.
def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def formatted(value, digits=2, suffix="", prefix="", compact=False):
    value = number(value)
    if value is None:
        return "—"
    if compact:
        for scale, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= scale:
                return f"{prefix}{value / scale:,.{digits}f}{unit}{suffix}"
    return f"{prefix}{value:,.{digits}f}{suffix}"


def percent(value):
    value = number(value)
    return formatted(value * 100 if value is not None else None, suffix="%")


def metric_grid(items, count=3):
    for offset in range(0, len(items), count):
        for col, (label, value) in zip(st.columns(count), items[offset:offset + count]):
            col.metric(label, value)


def monthly_bars(data):
    chart = data.copy().sort_values("Date")
    chart["Month"] = pd.to_datetime(chart["Date"]).dt.to_period("M")
    monthly = chart.groupby("Month", sort=True).agg(
        Date=("Date", "last"), Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"),
    )
    return monthly.reset_index(drop=True)


def trend_status(data, fast, slow):
    close = data["Close"].dropna()
    if len(close) < slow:
        return "Insufficient data"
    a = close.ewm(span=fast, min_periods=fast, adjust=False).mean().iloc[-1]
    b = close.ewm(span=slow, min_periods=slow, adjust=False).mean().iloc[-1]
    return "▲ Bullish" if a > b else "▼ Bearish" if a < b else "Neutral"


def historical_performance(data):
    close = data.set_index("Date")["Close"].dropna().sort_index()
    latest = close.iloc[-1]
    last_date = close.index[-1]
    def change(previous):
        return formatted((latest / previous - 1) * 100, suffix="%") if previous != 0 else "—"
    result = [("1 Day", change(close.iloc[-2]) if len(close) >= 2 else "—")]
    for label, cutoff in (
        ("1 Week", last_date - pd.DateOffset(weeks=1)),
        ("1 Month", last_date - pd.DateOffset(months=1)),
        ("3 Months", last_date - pd.DateOffset(months=3)),
        ("YTD", pd.Timestamp(last_date.year - 1, 12, 31)),
        ("1 Year", last_date - pd.DateOffset(years=1)),
    ):
        earlier = close.loc[close.index <= cutoff]
        result.append((label, change(earlier.iloc[-1]) if not earlier.empty else "—"))
    return result


@st.cache_data(ttl=900, show_spinner=False)
def get_dividend_history(symbol, start, end):
    try:
        history = yf.Ticker(symbol).history(start=start, end=end, actions=True, auto_adjust=False)
        if history.empty:
            return None
        if "Dividends" not in history:
            return None
        payments = history.loc[history["Dividends"] > 0, ["Dividends"]].copy()
        payments.index = pd.to_datetime(payments.index).tz_localize(None)
        return payments
    except Exception as exc:
        logger.warning("Dividend history failed for %s: %r", symbol, exc)
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_company_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.get_news(count=15) or []
    except Exception as exc:
        logger.warning("News failed for %s: %r", symbol, exc)
        return None


def news_fields(item):
    if not isinstance(item, dict):
        return None
    content = item.get("content") or item
    if not isinstance(content, dict):
        return None
    title = content.get("title")
    if not title:
        return None
    provider = content.get("provider") or {}
    publisher = provider.get("displayName") if isinstance(provider, dict) else None
    publisher = publisher or content.get("publisher") or "Unknown publisher"
    link = ""
    for candidate in (content.get("canonicalUrl"), content.get("clickThroughUrl"),
                      content.get("link"), item.get("link")):
        if isinstance(candidate, dict):
            candidate = candidate.get("url")
        if isinstance(candidate, str):
            try:
                parsed = urlparse(candidate)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    link = candidate
                    break
            except ValueError:
                pass
    raw_date = content.get("pubDate") or content.get("providerPublishTime")
    try:
        date = pd.to_datetime(raw_date, unit="s", utc=True) if isinstance(raw_date, (int, float)) else pd.to_datetime(raw_date, utc=True)
        date = date.tz_convert("Asia/Kuala_Lumpur").strftime("%Y-%m-%d %H:%M MYT") if pd.notna(date) else "Date unavailable"
    except (TypeError, ValueError, OverflowError):
        date = "Date unavailable"
    return str(title), str(publisher), date, link


currency = str(ticker_info.get("currency") or "")
price_prefix = f"{currency} " if currency else ""
monthly = monthly_bars(df1)

if backtest_view == "Overview":
    st.header("Stock Overview")
    latest = df1.iloc[-1]
    st.caption(f"Last available bar in selected date range: {pd.Timestamp(latest['Date']):%Y-%m-%d} · Historical prices follow the app's Yahoo Finance download settings; not a live quote.")
    previous = number(df1["Close"].iloc[-2]) if len(df1) > 1 else None
    close = number(latest["Close"])
    delta = formatted((close / previous - 1) * 100, suffix="%") if close is not None and previous not in (None, 0) else None
    cols = st.columns(3)
    cols[0].metric("Last Close", formatted(close, prefix=price_prefix), delta=delta)
    cols[1].metric("Day High", formatted(latest["High"], prefix=price_prefix))
    cols[2].metric("Day Low", formatted(latest["Low"], prefix=price_prefix))
    metric_grid([("Day Open", formatted(latest["Open"], prefix=price_prefix)), ("Volume", formatted(latest["Volume"], compact=True)), ("Currency", currency or "—")])
    st.subheader("Price Performance")
    metric_grid(historical_performance(df1))
    st.caption("Returns use the selected price history ending at its last available bar. Missing lookback history is shown as —.")
    st.subheader("Technical Trend Overview")
    trends = pd.DataFrame({
        "Indicator": ["EMA 10 / 20", "EMA 30 / 60"],
        **{label: [trend_status(data, 10, 20), trend_status(data, 30, 60)] for label, data in (("Daily", df1), ("Weekly", weekly), ("Monthly", monthly))},
    })
    st.dataframe(trends, hide_index=True, width="stretch")
    st.caption("EMA uses adjust=False, as in the backtest indicator. Current weekly/monthly bars may be incomplete. Trend is not a trade entry signal.")
    st.plotly_chart(plot_candlestick(df1, f"Daily Chart - {user_ticker}"), key="overview_chart")
    st.stop()

if backtest_view == "Monthly":
    st.header("Monthly Market Data")
    st.caption(f"{date_range_label} · Monthly OHLCV aggregated from the selected daily data. First/last months may be partial; dates show the last available trading day.")
    display_table(monthly, rows=24)
    chart = plot_candlestick(monthly, f"Monthly Chart - {user_ticker}")
    for period, color in ((30, "#A855F7"), (60, "#F97316")):
        chart.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Close"].ewm(span=period, min_periods=period).mean(), name=f"EMA {period}", line=dict(color=color)))
    st.plotly_chart(chart, key="monthly_candlestick_chart")
    metric_grid([("EMA 10 / 20", trend_status(monthly, 10, 20)), ("EMA 30 / 60", trend_status(monthly, 30, 60)), ("Monthly Bars", str(len(monthly)))])
    st.caption("Monthly is a chart and trend page. The original backtesting strategies remain available on Daily, Weekly and Summary.")
    st.stop()

if backtest_view == "Fundamental":
    st.header("Fundamental Analysis")
    st.caption("Latest available Yahoo Finance company snapshot, not historical data as of the selected End Date. Statement currency may differ from trading currency.")
    financial_currency = str(ticker_info.get("financialCurrency") or "")
    financial_prefix = f"{financial_currency} " if financial_currency else ""
    info = ticker_info
    groups = {
        "Valuation": [("Market Cap", formatted(info.get("marketCap"), prefix=price_prefix, compact=True)), ("P/E (TTM)", formatted(info.get("trailingPE"))), ("Forward P/E", formatted(info.get("forwardPE"))), ("Price / Book", formatted(info.get("priceToBook"))), ("EPS (TTM)", formatted(info.get("trailingEps"), prefix=price_prefix)), ("Trailing Annual Dividend Yield", percent(info.get("trailingAnnualDividendYield")))],
        "Growth & Profitability": [("Revenue (TTM)", formatted(info.get("totalRevenue"), prefix=financial_prefix, compact=True)), ("Revenue Growth", percent(info.get("revenueGrowth"))), ("Earnings Growth", percent(info.get("earningsGrowth"))), ("Return on Equity", percent(info.get("returnOnEquity"))), ("Return on Assets", percent(info.get("returnOnAssets"))), ("Profit Margin", percent(info.get("profitMargins")))],
        "Balance Sheet": [("Total Cash", formatted(info.get("totalCash"), prefix=financial_prefix, compact=True)), ("Total Debt", formatted(info.get("totalDebt"), prefix=financial_prefix, compact=True)), ("Debt / Equity", formatted(info.get("debtToEquity"), suffix="%"))],
    }
    if all(value == "—" for items in groups.values() for _, value in items):
        st.info("Fundamental metrics are currently unavailable for this instrument. Yahoo Finance may not provide company data or the request may have failed.")
    for heading, items in groups.items():
        st.subheader(heading)
        metric_grid(items)
    st.subheader("Company Profile")
    st.write(company_name)
    st.write(f"Sector: {info.get('sector') or '—'} · Industry: {info.get('industry') or '—'}")
    if info.get("longBusinessSummary"):
        st.write(str(info["longBusinessSummary"]))
    st.stop()

if backtest_view == "Dividend":
    st.header("Dividend History")
    st.caption(f"Cash distributions recorded during {date_range_label}. Amounts are per share, subject to Yahoo Finance coverage.")
    payments = get_dividend_history(user_ticker, user_input_start, download_end_date)
    if payments is None:
        st.info("Dividend data is unavailable. Please try again later or choose another stock.")
    elif payments.empty:
        st.info("No cash dividend records were returned for this date range.")
    else:
        annual = payments["Dividends"].groupby(payments.index.year).sum()
        metric_grid([("Total per Share (Selected Range)", formatted(payments["Dividends"].sum(), prefix=price_prefix, digits=4)), ("Latest Dividend per Share", formatted(payments["Dividends"].iloc[-1], prefix=price_prefix, digits=4)), ("Last Dividend Date", payments.index[-1].strftime("%Y-%m-%d"))])
        st.subheader("Annual Dividend per Share")
        st.caption("First and last years may be partial according to the selected date range.")
        st.plotly_chart(go.Figure(go.Bar(x=annual.index.astype(str), y=annual.values, name="Dividend per Share")), key="annual_dividend_chart")
        records = payments.rename_axis("Date").reset_index().sort_values("Date", ascending=False)
        st.dataframe(records, width="stretch", hide_index=True)
        st.download_button("Download Dividend CSV", records.to_csv(index=False).encode("utf-8"), file_name=f"{user_ticker}_dividends.csv", mime="text/csv")
    st.stop()

if backtest_view == "News":
    st.header("Company News")
    st.caption("Latest articles returned by Yahoo Finance; independent of the selected historical date range. Times shown in Malaysia time (MYT).")
    articles = get_company_news(user_ticker)
    seen = set()
    for article in articles or []:
        row = news_fields(article)
        if row is None:
            continue
        title, publisher, date, link = row
        if (title, link) in seen:
            continue
        seen.add((title, link))
        st.write(title)
        st.caption(f"{publisher} · {date}")
        if link:
            st.link_button("Read article", link)
        st.divider()
    if not seen:
        st.info("No news is currently available for this instrument, or the news request failed.")
    st.stop()


if backtest_view == "Daily":
    fig4 = plot_candlestick(
        df,
        f"Daily Chart - {user_ticker} ({date_range_label})",
    )
    st.plotly_chart(fig4, key="daily_candlestick_chart")

#####################################################################################

if backtest_view == "Weekly":
    st.markdown(f"### Weekly Market Data ({date_range_label})")
    display_table(weekly, rows=10)

    fig5 = plot_candlestick(
        weekly,
        f"Weekly Chart - {user_ticker} ({date_range_label})",
    )
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

# st.subheader('Daily Position')   
# display_table(daily_position, rows=1000)

# st.subheader('Weekly Position') 
# display_table(weekly_position, rows=1000)
# st.subheader('')
# st.subheader('')




#####################################################################################

st.subheader('')
st.subheader('')

def openHTML(test_html,width=900,height=800):
    file=codecs.open(test_html,'r')
    page = file.read()
    stc.html(page,width=width,height=height)

#BackTest

class EmaCross(SignalStrategy, TrailingStrategy):
    fast_period = 10
    slow_period = 20
    
    def init(self):
        super().init()
        
        ema1 = self.I(EMA, self.data.Close, self.fast_period)
        ema2 = self.I(EMA, self.data.Close, self.slow_period)

        # Where EMA 10 crosses EMA 20 upwards. Diff gives us [-1, 0, 1].
        signal = (pd.Series(ema1) > ema2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only

        entry_size = signal * .95

        self.set_signal(entry_size = entry_size)
        
        self.set_atr_periods(20)
        self.set_trailing_sl(2) #ATR stop loss (20*2)

      




#####################################################################################

class EmaCrossLongOnly(SignalStrategy):
    """Long-only EMA crossover with an EMA crossover exit."""

    fast_period = 10
    slow_period = 20

    def init(self):
        super().init()

        ema_fast = self.I(EMA, self.data.Close, self.fast_period)
        ema_slow = self.I(EMA, self.data.Close, self.slow_period)

        above_slow = (pd.Series(ema_fast) > ema_slow).astype(int)
        crossover_signal = above_slow.diff().fillna(0)

        # Buy only when EMA 10 crosses above EMA 20.
        entry_signal = (crossover_signal > 0).astype(float) * 0.95
        # Close the long position when EMA 20 crosses above EMA 10.
        exit_signal = (crossover_signal < 0).astype(float)

        self.set_signal(
            entry_size=entry_signal,
            exit_portion=exit_signal,
        )


class EmaCross30_60(EmaCross):
    """EMA 30/60 crossover with ATR trailing stop."""

    fast_period = 30
    slow_period = 60


class EmaCrossLongOnly30_60(EmaCrossLongOnly):
    """Long-only EMA 30/60 crossover with EMA crossover exit."""

    fast_period = 30
    slow_period = 60


if backtest_view in {"Daily", "Summary"}:
    if len(df1) < 61:
        st.info("Select a longer date range (at least 61 daily bars) to run all four strategies.")
        st.stop()
    daily_backtest_data = df1.set_index("Date").copy()
    daily_backtest_data.index = pd.to_datetime(daily_backtest_data.index)
    bt_daily = FractionalBacktest(daily_backtest_data,EmaCrossLongOnly,cash=1000, commission=0.002,trade_on_close=True)
    bt_daily_stats = bt_daily.run()
    print("Daily backtest results:")
    print(bt_daily_stats.to_string())

    bt_daily_atr = FractionalBacktest(daily_backtest_data,EmaCross,cash=1000, commission=0.002,trade_on_close=True)
    bt_daily_atr_stats = bt_daily_atr.run()
    print("Daily EMA + ATR backtest results:")
    print(bt_daily_atr_stats.to_string())

    bt_daily_30_60 = FractionalBacktest(
        daily_backtest_data,
        EmaCross30_60,
        cash=1000,
        commission=0.002,
        trade_on_close=True,
    )
    bt_daily_30_60_stats = bt_daily_30_60.run()
    print("Daily EMA 30/60 + ATR backtest results:")
    print(bt_daily_30_60_stats.to_string())

    bt_daily_long_30_60 = FractionalBacktest(
        daily_backtest_data,
        EmaCrossLongOnly30_60,
        cash=1000,
        commission=0.002,
        trade_on_close=True,
    )
    bt_daily_long_30_60_stats = bt_daily_long_30_60.run()
    print("Daily EMA 30/60 Exit results:")
    print(bt_daily_long_30_60_stats.to_string())


strategy_colors = {
    "Short-Term Trend + ATR Stop": "#3B82F6",
    "Long-Term Trend + ATR Stop": "#A855F7",
    "Short-Term Trend + EMA Exit": "#22C55E",
    "Long-Term Trend + EMA Exit": "#F97316",
}


def get_strategy_color(title):
    for strategy_name, color in strategy_colors.items():
        if strategy_name in title:
            return color
    return None


def display_colored_heading(title):
    color = get_strategy_color(title)
    if color is None:
        st.markdown(f"### {title}")
        return

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            margin: 1.25rem 0 0.55rem 0;
            padding: 0.25rem 0 0.25rem 0.75rem;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.25;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_strategy_legend():
    legend_items = "".join(
        f"""
        <span style="
            display: inline-block;
            margin: 0 0.45rem 0.45rem 0;
            padding: 0.35rem 0.6rem;
            border-radius: 0.4rem;
            border: 1px solid {color};
            color: {color};
            font-size: 0.82rem;
            font-weight: 600;
        ">{strategy_name}</span>
        """
        for strategy_name, color in strategy_colors.items()
    )
    st.markdown("#### Strategy Colour Guide")
    st.markdown(legend_items, unsafe_allow_html=True)


def display_trade_log(stats, title):
    trades = stats.get("_trades")
    if trades is None or trades.empty:
        st.info(f"No completed trades are available for {title}.")
        return

    trade_log = trades.copy()
    sort_columns = [
        column for column in ("EntryTime", "ExitTime") if column in trade_log.columns
    ]
    if sort_columns:
        trade_log = trade_log.sort_values(
            by=sort_columns,
            ascending=True,
            kind="stable",
        ).reset_index(drop=True)

    column_labels = {
        "EntryTime": "Entry Date",
        "ExitTime": "Exit Date",
        "Size": "Size",
        "EntryBar": "Entry Bar",
        "ExitBar": "Exit Bar",
        "EntryPrice": "Entry Price",
        "ExitPrice": "Exit Price",
        "SL": "Stop Loss",
        "TP": "Take Profit",
        "PnL": "P&L",
        "Commission": "Commission",
        "ReturnPct": "Return %",
        "Duration": "Duration",
    }
    preferred_columns = [
        "EntryTime",
        "ExitTime",
        "Size",
        "EntryBar",
        "ExitBar",
        "EntryPrice",
        "ExitPrice",
        "SL",
        "TP",
        "PnL",
        "Commission",
        "ReturnPct",
        "Duration",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in trade_log.columns
    ]
    trade_log = trade_log[ordered_columns].rename(columns=column_labels)

    for date_column in ("Entry Date", "Exit Date"):
        if date_column in trade_log.columns:
            trade_log[date_column] = pd.to_datetime(
                trade_log[date_column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    for price_column in (
        "Entry Price",
        "Exit Price",
        "Stop Loss",
        "Take Profit",
        "P&L",
        "Commission",
    ):
        if price_column in trade_log.columns:
            trade_log[price_column] = trade_log[price_column].map(
                lambda value: f"${value:,.2f}" if pd.notna(value) else "—"
            )
    if "Return %" in trade_log.columns:
        trade_log["Return %"] = trade_log["Return %"].map(
            lambda value: f"{value * 100:.2f}%" if pd.notna(value) else "—"
        )
    if "Size" in trade_log.columns:
        trade_log["Size"] = trade_log["Size"].map(
            lambda value: (
                f"{value:,.0f}"
                if pd.notna(value) and float(value).is_integer()
                else f"{value:,.4f}"
                if pd.notna(value)
                else "—"
            )
        )

    display_colored_heading(title)
    st.dataframe(trade_log, width="stretch", hide_index=True)


def display_special_objects(stats, title):
    equity_curve = stats.get("_equity_curve")
    display_colored_heading(f"{title} Equity Curve")
    if equity_curve is None or equity_curve.empty or "Equity" not in equity_curve:
        st.info("Equity curve is unavailable.")
        return

    st.line_chart(equity_curve[["Equity"]])

strategy_descriptions = {
    "Short-Term Trend + ATR Stop": (
        "Entry: EMA 10 crosses above EMA 20. "
        "Exit: 2 × ATR(20) trailing stop. Long-only."
    ),
    "Long-Term Trend + ATR Stop": (
        "Entry: EMA 30 crosses above EMA 60. "
        "Exit: 2 × ATR(20) trailing stop. Long-only."
    ),
    "Short-Term Trend + EMA Exit": (
        "Entry: EMA 10 crosses above EMA 20. "
        "Exit: EMA 10 crosses below EMA 20. Long-only."
    ),
    "Long-Term Trend + EMA Exit": (
        "Entry: EMA 30 crosses above EMA 60. "
        "Exit: EMA 30 crosses below EMA 60. Long-only."
    ),
}


def display_backtest_stats(stats, title):
    summary = stats.drop(
        labels=["_strategy", "_equity_curve", "_trades"],
        errors="ignore",
    )
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
    display_colored_heading(title)
    st.caption(strategy_descriptions.get(title, ""))

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


def display_strategy_comparison(results, title):
    metrics = [
        ("Return [%]", "Return"),
        ("Buy & Hold Return [%]", "Buy & Hold Return"),
        ("Equity Final [$]", "Final Equity"),
        ("CAGR [%]", "CAGR"),
        ("Max. Drawdown [%]", "Max Drawdown"),
        ("Sharpe Ratio", "Sharpe"),
        ("Sortino Ratio", "Sortino"),
        ("# Trades", "Trades"),
        ("Win Rate [%]", "Win Rate"),
        ("Profit Factor", "Profit Factor"),
    ]

    def format_comparison_value(metric, value):
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
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

    comparison = {}
    for strategy_name, stats in results.items():
        comparison[strategy_name] = {
            display_name: format_comparison_value(metric, stats.get(metric))
            for metric, display_name in metrics
        }

    comparison = pd.DataFrame(comparison)
    comparison.index.name = "Metric"
    st.markdown(f"### {title}")
    st.dataframe(comparison, width="stretch")


def display_summary_highlights(results, title):
    highlight_metrics = [
        ("Return [%]", "Best Return", True, lambda value: f"{value:.2f}%"),
        ("Sharpe Ratio", "Best Sharpe", True, lambda value: f"{value:.2f}"),
        (
            "Max. Drawdown [%]",
            "Lowest Drawdown",
            True,
            lambda value: f"{value:.2f}%",
        ),
    ]

    st.markdown(f"### {title} Highlights")
    columns = st.columns(3)
    for column, (metric, label, maximize, formatter) in zip(columns, highlight_metrics):
        candidates = []
        for strategy_name, stats in results.items():
            value = stats.get(metric)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                candidates.append((strategy_name, numeric_value))

        if not candidates:
            column.metric(label, "—")
            column.caption("No data")
            continue

        strategy_name, value = (
            max(candidates, key=lambda candidate: candidate[1])
            if maximize
            else min(candidates, key=lambda candidate: candidate[1])
        )
        column.metric(label, formatter(value))
        column.caption(strategy_name)


if backtest_view == "Daily":
    display_strategy_legend()
    display_backtest_stats(bt_daily_atr_stats, "Short-Term Trend + ATR Stop")
    display_trade_log(bt_daily_atr_stats, "Short-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_daily_atr_stats, "Short-Term Trend + ATR Stop")
    bt_daily_atr.plot(filename="EmaCrossATRDaily.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Short-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCrossATRDaily.html")

    display_backtest_stats(bt_daily_30_60_stats, "Long-Term Trend + ATR Stop")
    display_trade_log(bt_daily_30_60_stats, "Long-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_daily_30_60_stats, "Long-Term Trend + ATR Stop")
    bt_daily_30_60.plot(filename="EmaCross30_60ATRDaily.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Long-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCross30_60ATRDaily.html")

    display_backtest_stats(bt_daily_stats, "Short-Term Trend + EMA Exit")
    display_trade_log(bt_daily_stats, "Short-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_daily_stats, "Short-Term Trend + EMA Exit")
    bt_daily.plot(filename="EmaCrossLongOnly.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Short-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnly.html")

    display_backtest_stats(bt_daily_long_30_60_stats, "Long-Term Trend + EMA Exit")
    display_trade_log(bt_daily_long_30_60_stats, "Long-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_daily_long_30_60_stats, "Long-Term Trend + EMA Exit")
    bt_daily_long_30_60.plot(filename="EmaCrossLongOnly30_60Daily.html", open_browser=False)
    display_colored_heading(f"Daily Chart - Long-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnly30_60Daily.html")

    daily_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_daily_atr_stats,
        "Long-Term Trend + ATR Stop": bt_daily_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_daily_stats,
        "Long-Term Trend + EMA Exit": bt_daily_long_30_60_stats,
    }
    display_strategy_comparison(daily_strategy_results, "Daily Strategy Comparison")

#####################################################################################

if backtest_view in {"Weekly", "Summary"}:
    if len(weekly1) < 61:
        st.info("Select a longer date range (at least 61 weekly bars) to run all four strategies.")
        st.stop()
    bt_weekly = FractionalBacktest(weekly1, EmaCrossLongOnly,cash=1000, commission=0.002,trade_on_close=True)
    bt_weekly_stats = bt_weekly.run()
    print("Weekly backtest results:")
    print(bt_weekly_stats.to_string())

    bt_weekly_atr = FractionalBacktest(weekly1, EmaCross,cash=1000, commission=0.002,trade_on_close=True)
    bt_weekly_atr_stats = bt_weekly_atr.run()
    print("Weekly EMA + ATR backtest results:")
    print(bt_weekly_atr_stats.to_string())

    bt_weekly_30_60 = FractionalBacktest(
        weekly1,
        EmaCross30_60,
        cash=1000,
        commission=0.002,
        trade_on_close=True,
    )
    bt_weekly_30_60_stats = bt_weekly_30_60.run()
    print("Weekly EMA 30/60 + ATR backtest results:")
    print(bt_weekly_30_60_stats.to_string())

    bt_weekly_long_30_60 = FractionalBacktest(
        weekly1,
        EmaCrossLongOnly30_60,
        cash=1000,
        commission=0.002,
        trade_on_close=True,
    )
    bt_weekly_long_30_60_stats = bt_weekly_long_30_60.run()
    print("Weekly EMA 30/60 Exit results:")
    print(bt_weekly_long_30_60_stats.to_string())

if backtest_view == "Weekly":
    display_strategy_legend()
    display_backtest_stats(bt_weekly_atr_stats, "Short-Term Trend + ATR Stop")
    display_trade_log(bt_weekly_atr_stats, "Short-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_weekly_atr_stats, "Short-Term Trend + ATR Stop")
    bt_weekly_atr.plot(filename="EmaCrossATRWeekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Short-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCrossATRWeekly.html")

    display_backtest_stats(bt_weekly_30_60_stats, "Long-Term Trend + ATR Stop")
    display_trade_log(bt_weekly_30_60_stats, "Long-Term Trend + ATR Stop Trade Records")
    display_special_objects(bt_weekly_30_60_stats, "Long-Term Trend + ATR Stop")
    bt_weekly_30_60.plot(filename="EmaCross30_60ATRWeekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Long-Term Trend + ATR Stop ({date_range_label})")
    openHTML("EmaCross30_60ATRWeekly.html")

    display_backtest_stats(bt_weekly_stats, "Short-Term Trend + EMA Exit")
    display_trade_log(bt_weekly_stats, "Short-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_weekly_stats, "Short-Term Trend + EMA Exit")
    bt_weekly.plot(filename="EmaCrossLongOnlyWeekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Short-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnlyWeekly.html")

    display_backtest_stats(bt_weekly_long_30_60_stats, "Long-Term Trend + EMA Exit")
    display_trade_log(bt_weekly_long_30_60_stats, "Long-Term Trend + EMA Exit Trade Records")
    display_special_objects(bt_weekly_long_30_60_stats, "Long-Term Trend + EMA Exit")
    bt_weekly_long_30_60.plot(filename="EmaCrossLongOnly30_60Weekly.html", open_browser=False)
    display_colored_heading(f"Weekly Chart - Long-Term Trend + EMA Exit ({date_range_label})")
    openHTML("EmaCrossLongOnly30_60Weekly.html")

    weekly_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_weekly_atr_stats,
        "Long-Term Trend + ATR Stop": bt_weekly_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_weekly_stats,
        "Long-Term Trend + EMA Exit": bt_weekly_long_30_60_stats,
    }
    display_strategy_comparison(weekly_strategy_results, "Weekly Strategy Comparison")

if backtest_view == "Summary":
    st.markdown("## Daily & Weekly Strategy Dashboard")
    st.caption("Compare the four strategies across both timeframes.")

    daily_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_daily_atr_stats,
        "Long-Term Trend + ATR Stop": bt_daily_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_daily_stats,
        "Long-Term Trend + EMA Exit": bt_daily_long_30_60_stats,
    }
    weekly_strategy_results = {
        "Short-Term Trend + ATR Stop": bt_weekly_atr_stats,
        "Long-Term Trend + ATR Stop": bt_weekly_30_60_stats,
        "Short-Term Trend + EMA Exit": bt_weekly_stats,
        "Long-Term Trend + EMA Exit": bt_weekly_long_30_60_stats,
    }
    display_summary_highlights(daily_strategy_results, "Daily")
    display_summary_highlights(weekly_strategy_results, "Weekly")
    display_strategy_comparison(daily_strategy_results, "Daily Strategy Comparison")
    display_strategy_comparison(weekly_strategy_results, "Weekly Strategy Comparison")
