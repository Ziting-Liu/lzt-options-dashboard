"""
calculations.py
----------------
Pure math / scoring functions. No network calls here — everything takes
already-fetched data (price history, financial statements) and returns
numbers or scores. Keeping this separate from data_fetch.py makes it easy
to unit test and to reuse once we build the Options Dashboard.
"""

import numpy as np
import pandas as pd


# ============================================================
# PRICE-HISTORY BASED METRICS
# ============================================================

def cagr(price_series: pd.Series, years: float):
    """Compound annual growth rate over the given number of years, using
    the earliest and latest available closes within that lookback."""
    if price_series is None or price_series.empty:
        return None
    cutoff = price_series.index.max() - pd.Timedelta(days=int(years * 365.25))
    window = price_series[price_series.index >= cutoff]
    if len(window) < 2:
        return None
    start, end = float(window.iloc[0]), float(window.iloc[-1])
    actual_years = (window.index[-1] - window.index[0]).days / 365.25
    if start <= 0 or actual_years <= 0:
        return None
    return (end / start) ** (1 / actual_years) - 1


def max_drawdown(price_series: pd.Series):
    """Largest peak-to-trough decline over the given series."""
    if price_series is None or price_series.empty:
        return None
    running_max = price_series.cummax()
    drawdown = (price_series - running_max) / running_max
    return float(drawdown.min())


def best_worst_calendar_year(price_series: pd.Series):
    """Returns (best_year, best_return, worst_year, worst_return) using
    calendar-year close-to-close returns."""
    if price_series is None or price_series.empty:
        return None, None, None, None
    yearly = price_series.resample("YE").last()
    yearly_returns = yearly.pct_change().dropna()
    if yearly_returns.empty:
        return None, None, None, None
    best_year = yearly_returns.idxmax().year
    worst_year = yearly_returns.idxmin().year
    return best_year, float(yearly_returns.max()), worst_year, float(yearly_returns.min())


def rolling_return(price_series: pd.Series, years: int):
    """Average annualized rolling return over N-year windows across history."""
    if price_series is None or price_series.empty:
        return None
    window_days = int(years * 365.25)
    monthly = price_series.resample("ME").last().dropna()
    step = max(1, round(window_days / 30))
    if len(monthly) <= step:
        return None
    rets = (monthly.shift(-step) / monthly) ** (1 / years) - 1
    rets = rets.dropna()
    return float(rets.mean()) if not rets.empty else None


def beta_vs_benchmark(stock_prices: pd.Series, bench_prices: pd.Series):
    """Simple daily-return beta of stock vs. benchmark (e.g. SPY)."""
    if stock_prices is None or bench_prices is None:
        return None
    df = pd.DataFrame({"s": stock_prices, "b": bench_prices}).dropna()
    if len(df) < 30:
        return None
    stock_ret = df["s"].pct_change().dropna()
    bench_ret = df["b"].pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 30 or aligned.iloc[:, 1].var() == 0:
        return None
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    return float(cov / var)


def historical_volatility(price_series: pd.Series, window: int = 30):
    """Annualized historical volatility using daily log returns."""
    if price_series is None or len(price_series) < window + 1:
        return None
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    return float(log_ret.tail(window).std() * np.sqrt(252))


def relative_strength_vs_spy(stock_prices: pd.Series, spy_prices: pd.Series, days: int = 63):
    """Stock's return minus SPY's return over the trailing window (positive = outperforming)."""
    if stock_prices is None or spy_prices is None:
        return None
    df = pd.DataFrame({"s": stock_prices, "b": spy_prices}).dropna()
    if len(df) < days + 1:
        return None
    window = df.tail(days + 1)
    stock_ret = window["s"].iloc[-1] / window["s"].iloc[0] - 1
    bench_ret = window["b"].iloc[-1] / window["b"].iloc[0] - 1
    return float(stock_ret - bench_ret)


def distance_pct(current, reference):
    """% distance of current price from a reference level (MA, 52wk high/low, etc)."""
    if current is None or reference in (None, 0) or pd.isna(reference):
        return None
    return (current - reference) / reference * 100


def pe_percentile_approx(price_series: pd.Series, trailing_eps_history: pd.Series, current_pe):
    """
    Approximates a historical PE percentile using price / most-recent-known-EPS-at-each-date.
    This is a simplification (real percentile tools use PE-at-each-date using the EPS known
    at that time) but works reasonably well with the ~4-5 years of quarterly EPS yfinance
    exposes for free. Returns a 0-100 percentile of where current_pe sits vs. its own history.
    """
    if price_series is None or trailing_eps_history is None or trailing_eps_history.empty or current_pe is None:
        return None
    eps_sorted = trailing_eps_history.sort_index()
    aligned_eps = eps_sorted.reindex(price_series.index, method="ffill")
    hist_pe = (price_series / aligned_eps).replace([np.inf, -np.inf], np.nan).dropna()
    hist_pe = hist_pe[hist_pe > 0]
    if hist_pe.empty:
        return None
    return float((hist_pe < current_pe).mean() * 100)


# ============================================================
# FUNDAMENTAL GROWTH / MARGIN HELPERS
# ============================================================

def yoy_growth(series: pd.Series):
    """Most recent year-over-year growth rate from an annual series (already sorted ascending)."""
    if series is None or len(series.dropna()) < 2:
        return None
    s = series.dropna().sort_index()
    prev, curr = s.iloc[-2], s.iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(curr):
        return None
    return (curr / prev) - 1


def cagr_from_series(series: pd.Series, years: int):
    """CAGR computed from an annual fundamentals series (e.g. revenue by fiscal year)."""
    if series is None:
        return None
    s = series.dropna().sort_index()
    if len(s) < 2:
        return None
    s = s.tail(years + 1)
    if len(s) < 2:
        return None
    start, end = s.iloc[0], s.iloc[-1]
    n = len(s) - 1
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / n) - 1


def safe_ratio(numerator, denominator):
    try:
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return None


# ============================================================
# RULE-BASED BUSINESS QUALITY SCORE
# ============================================================

def score_to_stars(score_0_to_5: float) -> str:
    """Renders a 0-5 float score as ⭐ text, half-stars shown as a filled star
    (keeps it simple/legible rather than trying to render half-star glyphs)."""
    if score_0_to_5 is None:
        return "N/A"
    full = int(round(score_0_to_5))
    full = max(0, min(5, full))
    return "⭐" * full + "☆" * (5 - full)


def _clamp_score(value, low, high, invert=False):
    """Maps a raw metric onto a 0-5 score by linear interpolation between low/high."""
    if value is None or pd.isna(value):
        return None
    if invert:
        value = -value
        low, high = -high, -low
    if high == low:
        return None
    frac = (value - low) / (high - low)
    frac = max(0.0, min(1.0, frac))
    return round(frac * 5, 1)


def compute_quality_scores(metrics: dict):
    """
    Rule-based (no AI) scoring across 5 categories, each 0-5 stars, plus an
    overall 0-10 score. Thresholds are intentionally conservative/long-term
    oriented per the project philosophy (reduce emotional decisions, not
    chase momentum). All thresholds are adjustable constants below.
    """
    scores = {}

    # --- Business Quality: profitability + capital efficiency ---
    roic = metrics.get("roic")
    gross_margin = metrics.get("gross_margin")
    net_margin = metrics.get("net_margin")
    business_components = [
        _clamp_score(roic, 0.05, 0.20),
        _clamp_score(gross_margin, 0.20, 0.60),
        _clamp_score(net_margin, 0.05, 0.25),
    ]
    business_components = [c for c in business_components if c is not None]
    scores["Business Quality"] = round(sum(business_components) / len(business_components), 1) if business_components else None

    # --- Valuation: lower PE percentile / PEG is better (inverted) ---
    pe_percentile = metrics.get("pe_percentile")
    peg = metrics.get("peg_ratio")
    valuation_components = [
        _clamp_score(pe_percentile, 0, 100, invert=True),
        _clamp_score(peg, 0.5, 3.0, invert=True) if peg else None,
    ]
    valuation_components = [c for c in valuation_components if c is not None]
    scores["Valuation"] = round(sum(valuation_components) / len(valuation_components), 1) if valuation_components else None

    # --- Growth ---
    rev_growth = metrics.get("revenue_growth_yoy")
    eps_growth = metrics.get("eps_growth_yoy")
    growth_components = [
        _clamp_score(rev_growth, 0.0, 0.20),
        _clamp_score(eps_growth, 0.0, 0.20),
    ]
    growth_components = [c for c in growth_components if c is not None]
    scores["Growth"] = round(sum(growth_components) / len(growth_components), 1) if growth_components else None

    # --- Financial Strength ---
    debt_equity = metrics.get("debt_to_equity")
    current_ratio = metrics.get("current_ratio")
    interest_coverage = metrics.get("interest_coverage")
    strength_components = [
        _clamp_score(debt_equity, 0.0, 2.0, invert=True),
        _clamp_score(current_ratio, 1.0, 2.5),
        _clamp_score(interest_coverage, 2.0, 10.0),
    ]
    strength_components = [c for c in strength_components if c is not None]
    scores["Financial Strength"] = round(sum(strength_components) / len(strength_components), 1) if strength_components else None

    # --- Options Premium potential (placeholder until Options Dashboard ships) ---
    hv = metrics.get("historical_volatility_30d")
    scores["Options Premium"] = _clamp_score(hv, 0.15, 0.55) if hv is not None else None

    # --- Overall (weighted average of whatever categories we could compute) ---
    weights = {
        "Business Quality": 0.30,
        "Valuation": 0.20,
        "Growth": 0.20,
        "Financial Strength": 0.20,
        "Options Premium": 0.10,
    }
    weighted_sum, weight_total = 0.0, 0.0
    for category, weight in weights.items():
        val = scores.get(category)
        if val is not None:
            weighted_sum += val * weight
            weight_total += weight
    overall = round((weighted_sum / weight_total) * 2, 1) if weight_total > 0 else None  # scale 0-5 -> 0-10

    return scores, overall
