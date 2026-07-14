"""
data_fetch.py
--------------
All network/yfinance calls live here, each wrapped in st.cache_data so the
UI stays fast and we don't hammer Yahoo Finance on every widget interaction.
Every function degrades gracefully (returns None/empty rather than raising)
so a missing field for one ticker never crashes the whole dashboard.
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600)
def get_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_financials(ticker: str):
    """Returns (income_stmt, balance_sheet, cashflow) as annual DataFrames (columns = fiscal years)."""
    try:
        t = yf.Ticker(ticker)
        income = t.financials
        balance = t.balance_sheet
        cashflow = t.cashflow
        return income, balance, cashflow
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def get_quarterly_eps_history(ticker: str) -> pd.Series:
    """Trailing EPS by quarter end date, used for the PE-percentile approximation."""
    try:
        t = yf.Ticker(ticker)
        q_income = t.quarterly_financials
        if q_income is None or q_income.empty:
            return pd.Series(dtype=float)
        # Diluted EPS row name varies by ticker/version; try a few common labels.
        for label in ["Diluted EPS", "Basic EPS"]:
            if label in q_income.index:
                s = q_income.loc[label].dropna()
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def get_price_history(ticker: str, years: int = 20) -> pd.DataFrame:
    """Long daily price history for CAGR / drawdown / beta / MA calculations."""
    try:
        start = datetime.today() - timedelta(days=years * 365 + 30)
        df = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_earnings_calendar(ticker: str):
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if isinstance(dates, list) and dates:
                return dates[0]
        return None
    except Exception:
        return None


def days_until(date_value):
    if date_value is None:
        return None
    try:
        target = pd.to_datetime(date_value).date()
        return (target - datetime.today().date()).days
    except Exception:
        return None
