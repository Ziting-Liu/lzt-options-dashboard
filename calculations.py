"""
calculations.py
----------------
Pure math / scoring functions. No network calls here — everything takes
already-fetched data (price history, financial statements) and returns
numbers or scores. Keeping this separate from data_fetch.py makes it easy
to unit test and to reuse once we build the Options Dashboard.
"""

import math

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
    Approximates a historical PE percentile using price / trailing-twelve-month EPS
    at each date. This is a simplification (real percentile tools use PE-at-each-date
    using the EPS known at that time) but works reasonably well with the ~4-5 years
    of quarterly EPS yfinance exposes for free. Returns a 0-100 percentile of where
    current_pe sits vs. its own history.

    Two adjustments guard against cyclical/commodity businesses (energy, shipping,
    materials, etc.) whose lumpy quarterly earnings would otherwise wreck this metric:
      1. We sum trailing 4 quarters (TTM) instead of using a single quarter's EPS,
         so one unusually weak or seasonal quarter doesn't spike the ratio.
      2. We winsorize extreme outliers (a near-zero-earnings quarter can still send
         PE to absurd multiples even on a TTM basis) rather than letting a few
         freak data points dominate the whole percentile.
    """
    if price_series is None or trailing_eps_history is None or trailing_eps_history.empty or current_pe is None:
        return None
    ttm_eps = trailing_eps_history.sort_index().rolling(4, min_periods=4).sum().dropna()
    if ttm_eps.empty:
        return None
    aligned_eps = ttm_eps.reindex(price_series.index, method="ffill")
    hist_pe = (price_series / aligned_eps).replace([np.inf, -np.inf], np.nan).dropna()
    hist_pe = hist_pe[hist_pe > 0]
    if hist_pe.empty:
        return None
    # Winsorize: drop readings more than 5x the median — these come from
    # near-zero-earnings quarters, not genuine valuation extremes.
    median_pe = hist_pe.median()
    if median_pe > 0:
        hist_pe = hist_pe[hist_pe <= median_pe * 5]
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
# OPTIONS MATH (Black-Scholes) — conservative, long-term-investor framing
# ============================================================
# These are simplified Black-Scholes estimates for decision support, not
# a market-maker-grade pricing engine. Real option prices reflect the full
# order book (bid/ask spread, skew, early-exercise for American options),
# so treat "suggested strike" / "probability of assignment" as a reasonable
# planning estimate, not a guarantee.

RISK_FREE_RATE = 0.045  # approx short-term T-bill yield; adjust if it drifts meaningfully

# Conservative targets: low delta = low probability of assignment, favors
# the "reduce emotional decisions, don't chase premium" philosophy.
TARGET_DELTA_CSP = -0.20
TARGET_DELTA_CC = 0.20
TARGET_DTE_DAYS = 35  # the classic ~30-45 DTE sweet spot for theta decay vs. gamma risk


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def bs_delta(spot, strike, dte_days, iv, option_type="put", r=RISK_FREE_RATE):
    """Black-Scholes delta. option_type is 'call' or 'put'."""
    if spot is None or strike is None or iv is None or dte_days is None or dte_days <= 0 or iv <= 0:
        return None
    t = dte_days / 365.0
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    if option_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1  # put delta is negative


def pick_expiration_near_target(available_dtes: list, target_days: int = TARGET_DTE_DAYS):
    """Given a list of available DTEs (ints), returns the one closest to the target."""
    if not available_dtes:
        return None
    return min(available_dtes, key=lambda d: abs(d - target_days))


def pick_conservative_strike(chain_df, spot, dte_days, option_type="put", target_delta=None):
    """
    Scans an options chain DataFrame (must have 'strike' and 'impliedVolatility'
    columns, as yfinance's option_chain() provides) and returns the row whose
    Black-Scholes delta is closest to the conservative target — i.e. the strike
    with a low, deliberate probability of assignment rather than the richest premium.
    """
    if chain_df is None or chain_df.empty:
        return None
    target = target_delta if target_delta is not None else (TARGET_DELTA_CSP if option_type == "put" else TARGET_DELTA_CC)

    best_row, best_diff, best_delta = None, None, None
    for _, row in chain_df.iterrows():
        iv = row.get("impliedVolatility")
        strike = row.get("strike")
        # Only consider genuinely out-of-the-money strikes: puts below spot, calls above spot
        if option_type == "put" and strike >= spot:
            continue
        if option_type == "call" and strike <= spot:
            continue
        delta = bs_delta(spot, strike, dte_days, iv, option_type)
        if delta is None:
            continue
        diff = abs(delta - target)
        if best_diff is None or diff < best_diff:
            best_diff, best_row, best_delta = diff, row, delta
    if best_row is None:
        return None
    return {
        "strike": float(best_row["strike"]),
        "delta": best_delta,
        "iv": float(best_row.get("impliedVolatility")) if best_row.get("impliedVolatility") is not None else None,
        "bid": float(best_row.get("bid", 0) or 0),
        "ask": float(best_row.get("ask", 0) or 0),
        "last_price": float(best_row.get("lastPrice", 0) or 0),
    }


def estimate_premium(option_row: dict):
    """Uses mid of bid/ask when both are quoted; falls back to lastPrice otherwise."""
    if option_row is None:
        return None
    bid, ask, last = option_row.get("bid", 0), option_row.get("ask", 0), option_row.get("last_price", 0)
    if bid and ask:
        return (bid + ask) / 2
    return last or None


def annualized_yield(premium, basis_price, dte_days):
    """Simple annualized yield: premium / capital-at-risk, scaled to a full year."""
    if premium is None or basis_price in (None, 0) or dte_days in (None, 0):
        return None
    return (premium / basis_price) * (365 / dte_days)


def expected_move(spot, atm_iv, dte_days):
    """Approximate 1-standard-deviation expected move by expiration."""
    if spot is None or atm_iv is None or dte_days is None:
        return None
    return spot * atm_iv * np.sqrt(dte_days / 365.0)


def find_atm_iv(calls_df, puts_df, spot):
    """Averages the call and put implied volatility at the strike closest to spot."""
    if calls_df is None or calls_df.empty:
        return None
    calls_df = calls_df.copy()
    calls_df["dist"] = (calls_df["strike"] - spot).abs()
    atm_call = calls_df.sort_values("dist").iloc[0]
    ivs = [atm_call.get("impliedVolatility")]
    if puts_df is not None and not puts_df.empty:
        puts_df = puts_df.copy()
        puts_df["dist"] = (puts_df["strike"] - spot).abs()
        atm_put = puts_df.sort_values("dist").iloc[0]
        if atm_put.get("impliedVolatility"):
            ivs.append(atm_put.get("impliedVolatility"))
    ivs = [v for v in ivs if v is not None and v > 0]
    return float(np.mean(ivs)) if ivs else None


def iv_rank_and_percentile(current_iv, history_values: list):
    """
    Given a list of past ATM IV readings (any order) plus the current value,
    returns (iv_rank, iv_percentile) — both 0-100. Needs at least a handful
    of historical points to be meaningful.
    """
    if current_iv is None or not history_values:
        return None, None
    values = list(history_values) + [current_iv]
    lo, hi = min(values), max(values)
    iv_rank = ((current_iv - lo) / (hi - lo) * 100) if hi > lo else 50.0
    iv_percentile = float(np.mean([v <= current_iv for v in values]) * 100)
    return iv_rank, iv_percentile


def iv_rank_proxy_from_hv(price_series: pd.Series, current_iv, window: int = 30, lookback_days: int = 252):
    """
    Fallback used until enough real daily IV snapshots have been logged:
    builds a rolling historical-volatility series over the past ~1 year and
    finds where current ATM IV would rank against that HV distribution.
    This is a commonly used stand-in but is NOT the same as true IV Rank
    (which requires actual past option IV, not realized price volatility).
    """
    if price_series is None or current_iv is None or len(price_series) < window + lookback_days:
        # Not enough price history for a full rolling year — use what's available.
        if price_series is None or len(price_series) < window + 20:
            return None, None
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    rolling_hv = log_ret.rolling(window).std() * np.sqrt(252)
    rolling_hv = rolling_hv.dropna().tail(lookback_days)
    if rolling_hv.empty:
        return None, None
    return iv_rank_and_percentile(current_iv, rolling_hv.tolist())


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


# Sector groupings used to adjust scoring thresholds below. yfinance's info.get("sector")
# returns GICS-style sector names; these are the ones that matter for our adjustments.
CAPITAL_INTENSIVE_SECTORS = {"Energy", "Utilities", "Real Estate", "Industrials", "Basic Materials", "Materials"}
FINANCIAL_SECTORS = {"Financial Services", "Financials", "Financial"}


def compute_quality_scores(metrics: dict, sector: str = None):
    """
    Rule-based (no AI) scoring across 5 categories, each 0-5 stars, plus an
    overall 0-10 score. Thresholds are intentionally conservative/long-term
    oriented per the project philosophy (reduce emotional decisions, not
    chase momentum).

    IMPORTANT: thresholds are sector-adjusted for capital-intensive industries
    (energy, utilities, industrials, real estate, materials), which structurally
    carry more debt and thinner margins than asset-light businesses (software,
    services) because their business model requires large, debt-financed physical
    assets against long-term contracts. Without this adjustment, the scorer would
    systematically penalize an entire category of otherwise-healthy businesses
    just for looking like a bank/utility/pipeline operator instead of a software
    company — that's not a sign of lower quality, it's a different business model.
    Financials are excluded from the debt/equity check entirely, since leverage
    *is* the banking/insurance business model and a D/E ratio there measures
    something fundamentally different than it does for an operating company.
    """
    is_capital_intensive = sector in CAPITAL_INTENSIVE_SECTORS
    is_financial = sector in FINANCIAL_SECTORS

    scores = {}

    # --- Business Quality: profitability + capital efficiency ---
    roic = metrics.get("roic")
    gross_margin = metrics.get("gross_margin")
    net_margin = metrics.get("net_margin")

    if is_capital_intensive:
        roic_range, gross_range, net_range = (0.04, 0.12), (0.10, 0.40), (0.03, 0.15)
    else:
        roic_range, gross_range, net_range = (0.05, 0.20), (0.20, 0.60), (0.05, 0.25)

    business_components = [
        _clamp_score(roic, *roic_range),
        _clamp_score(gross_margin, *gross_range),
        _clamp_score(net_margin, *net_range),
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
    # Capital-intensive / mature sectors don't reasonably grow at software-style rates;
    # cap the "excellent" end of the range lower so a steady, mature grower isn't
    # scored as if it failed to grow like a tech company.
    growth_range = (0.0, 0.10) if is_capital_intensive else (0.0, 0.20)
    growth_components = [
        _clamp_score(rev_growth, *growth_range),
        _clamp_score(eps_growth, *growth_range),
    ]
    growth_components = [c for c in growth_components if c is not None]
    scores["Growth"] = round(sum(growth_components) / len(growth_components), 1) if growth_components else None

    # --- Financial Strength ---
    debt_equity = metrics.get("debt_to_equity")
    current_ratio = metrics.get("current_ratio")
    interest_coverage = metrics.get("interest_coverage")

    strength_components = []
    if not is_financial:
        de_range = (0.0, 4.0) if is_capital_intensive else (0.0, 2.0)
        strength_components.append(_clamp_score(debt_equity, *de_range, invert=True))
    strength_components.append(_clamp_score(current_ratio, 1.0, 2.5))
    strength_components.append(_clamp_score(interest_coverage, 2.0, 10.0))
    strength_components = [c for c in strength_components if c is not None]
    scores["Financial Strength"] = round(sum(strength_components) / len(strength_components), 1) if strength_components else None

    # --- Options Premium potential ---
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
