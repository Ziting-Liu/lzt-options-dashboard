import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import data_fetch as df_
import calculations as calc
import storage

st.set_page_config(page_title="LZT Investment Dashboard", layout="wide", page_icon="📈")

# Colorblind-safe palette (Okabe-Ito) — used everywhere instead of red/green.
COLOR_UP = "#0072B2"    # blue = positive / good
COLOR_DOWN = "#E69F00"  # orange = negative / caution
COLOR_NEUTRAL = "#666666"


# ============================================================
# FORMATTING HELPERS
# ============================================================
def fmt_money(x, decimals=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    return f"${x:,.{decimals}f}"


def fmt_big(x):
    """Formats large numbers as B/T (market cap, enterprise value, etc)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    x = float(x)
    if abs(x) >= 1e12:
        return f"${x / 1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"${x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x:,.0f}"


def fmt_pct(x, decimals=1):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def signed_pct_html(x, decimals=1, label=""):
    """Renders a % value with an arrow + colorblind-safe color instead of red/green."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return f"<span style='color:{COLOR_NEUTRAL}'>N/A</span>"
    arrow = "▲" if x >= 0 else "▼"
    color = COLOR_UP if x >= 0 else COLOR_DOWN
    prefix = f"{label} " if label else ""
    return f"<span style='color:{color};font-weight:600'>{prefix}{arrow} {abs(x) * 100:.{decimals}f}%</span>"


def get_row(frame: pd.DataFrame, labels: list):
    """Finds the first matching row (by label) in a yfinance financial statement DataFrame,
    trying several possible label variants since these change across yfinance versions."""
    if frame is None or frame.empty:
        return None
    for label in labels:
        if label in frame.index:
            s = frame.loc[label].dropna()
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return None


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📈 LZT Dashboard")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
st.sidebar.caption("Decision-support for long-term investing + conservative options selling. Not a trading signal tool.")

if not ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

with st.spinner(f"Loading {ticker}..."):
    info = df_.get_info(ticker)
    income, balance, cashflow = df_.get_financials(ticker)
    eps_hist = df_.get_quarterly_eps_history(ticker)
    prices = df_.get_price_history(ticker, years=21)
    spy_prices = df_.get_price_history("SPY", years=21)
    earnings_date = df_.get_earnings_calendar(ticker)

if prices.empty:
    st.error(f"No price data found for '{ticker}'. Check the ticker symbol.")
    st.stop()

close = prices["Close"]
spy_close = spy_prices["Close"] if not spy_prices.empty else None
current_price = float(close.iloc[-1])

st.title(f"{info.get('shortName', ticker)} ({ticker})")

tabs = st.tabs([
    "Overview", "Valuation", "Growth", "Profitability",
    "Financial Strength", "Technicals", "Options Dashboard",
    "Historical Performance", "Quality Score", "Journal",
])

# ============================================================
# 1. COMPANY OVERVIEW
# ============================================================
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Sector", info.get("sector", "N/A"))
    c2.metric("Industry", info.get("industry", "N/A"))
    c3.metric("Market Cap", fmt_big(info.get("marketCap")))

    c4, c5, c6 = st.columns(3)
    c4.metric("Enterprise Value", fmt_big(info.get("enterpriseValue")))
    c5.metric("Current Price", fmt_money(current_price))
    c6.metric("Dividend Yield", fmt_pct(info.get("dividendYield", 0) / 100 if info.get("dividendYield") else None))

    c7, c8, c9 = st.columns(3)
    c7.metric("52-Week High", fmt_money(info.get("fiftyTwoWeekHigh")))
    c8.metric("52-Week Low", fmt_money(info.get("fiftyTwoWeekLow")))
    days_to_earn = df_.days_until(earnings_date)
    c9.metric("Next Earnings", str(earnings_date)[:10] if earnings_date else "N/A",
              delta=f"{days_to_earn} days" if days_to_earn is not None else None,
              delta_color="off")

    ex_div = info.get("exDividendDate")
    if ex_div:
        try:
            ex_div = datetime.fromtimestamp(ex_div).strftime("%Y-%m-%d")
        except Exception:
            pass
    st.metric("Ex-Dividend Date", str(ex_div) if ex_div else "N/A")

# ============================================================
# 2. VALUATION
# ============================================================
with tabs[1]:
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")

    close_5y = close[close.index >= (close.index.max() - pd.Timedelta(days=5 * 365))]
    close_10y = close[close.index >= (close.index.max() - pd.Timedelta(days=10 * 365))]
    pe_percentile_5y = calc.pe_percentile_approx(close_5y, eps_hist, trailing_pe)
    pe_percentile_10y = calc.pe_percentile_approx(close_10y, eps_hist, trailing_pe)

    c1, c2, c3 = st.columns(3)
    c1.metric("Trailing P/E", f"{trailing_pe:.2f}" if trailing_pe else "N/A")
    c2.metric("Forward P/E", f"{forward_pe:.2f}" if forward_pe else "N/A")
    c3.metric("PEG Ratio", f"{info.get('pegRatio'):.2f}" if info.get("pegRatio") else "N/A")

    c4, c5, c6 = st.columns(3)
    c4.metric("EV / EBITDA", f"{info.get('enterpriseToEbitda'):.2f}" if info.get("enterpriseToEbitda") else "N/A")
    c5.metric("Price / Sales", f"{info.get('priceToSalesTrailing12Months'):.2f}" if info.get("priceToSalesTrailing12Months") else "N/A")
    fcf_info = info.get("freeCashflow")
    p_fcf = calc.safe_ratio(info.get("marketCap"), fcf_info)
    c6.metric("Price / FCF", f"{p_fcf:.2f}" if p_fcf else "N/A")

    st.markdown("**Historical P/E Percentile** — where today's P/E sits vs. its own recent history (lower = cheaper than usual).")
    c7, c8 = st.columns(2)
    c7.metric("5-Year P/E Percentile", f"{pe_percentile_5y:.0f}th" if pe_percentile_5y is not None else "N/A")
    c8.metric("10-Year P/E Percentile", f"{pe_percentile_10y:.0f}th" if pe_percentile_10y is not None else "N/A")
    st.caption(
        "⚠️ Approximated from ~4-5 years of quarterly EPS available via free data — "
        "treat as directional context, not a precise historical percentile."
    )

# ============================================================
# 3. GROWTH
# ============================================================
with tabs[2]:
    revenue = get_row(income, ["Total Revenue", "TotalRevenue"])
    net_income = get_row(income, ["Net Income", "NetIncome"])
    eps_annual = get_row(income, ["Diluted EPS", "DilutedEPS", "Basic EPS"])
    fcf_row = get_row(cashflow, ["Free Cash Flow", "FreeCashFlow"])
    if fcf_row is None:
        ocf = get_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = get_row(cashflow, ["Capital Expenditure", "CapitalExpenditures"])
        if ocf is not None and capex is not None:
            fcf_row = (ocf + capex).dropna()  # capex is typically negative already

    rev_growth = calc.yoy_growth(revenue)
    eps_growth = calc.yoy_growth(eps_annual) if eps_annual is not None else calc.yoy_growth(net_income)
    fcf_growth = calc.yoy_growth(fcf_row)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Revenue Growth (YoY)**<br>{signed_pct_html(rev_growth)}", unsafe_allow_html=True)
    c2.markdown(f"**EPS Growth (YoY)**<br>{signed_pct_html(eps_growth)}", unsafe_allow_html=True)
    c3.markdown(f"**FCF Growth (YoY)**<br>{signed_pct_html(fcf_growth)}", unsafe_allow_html=True)

    st.markdown("---")
    rev_cagr_3y = calc.cagr_from_series(revenue, 3)
    rev_cagr_5y = calc.cagr_from_series(revenue, 5)
    eps_cagr_3y = calc.cagr_from_series(eps_annual, 3) if eps_annual is not None else None
    eps_cagr_5y = calc.cagr_from_series(eps_annual, 5) if eps_annual is not None else None

    c4, c5, c6, c7 = st.columns(4)
    c4.markdown(f"**3Y Revenue CAGR**<br>{signed_pct_html(rev_cagr_3y)}", unsafe_allow_html=True)
    c5.markdown(f"**5Y Revenue CAGR**<br>{signed_pct_html(rev_cagr_5y)}", unsafe_allow_html=True)
    c6.markdown(f"**3Y EPS CAGR**<br>{signed_pct_html(eps_cagr_3y)}", unsafe_allow_html=True)
    c7.markdown(f"**5Y EPS CAGR**<br>{signed_pct_html(eps_cagr_5y)}", unsafe_allow_html=True)

    st.caption("Based on annual statements yfinance provides (typically ~4 fiscal years of history).")

# ============================================================
# 4. PROFITABILITY
# ============================================================
with tabs[3]:
    gross_margin = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")
    net_margin = info.get("profitMargins")
    equity = get_row(balance, ["Stockholders Equity", "Total Stockholder Equity"])
    total_debt_row = get_row(balance, ["Total Debt"])
    cash_row = get_row(balance, [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash",
    ])

    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")

    # ROIC approximation: NOPAT / (Debt + Equity - Cash), using most recent fiscal year
    roic = None
    op_income_row = get_row(income, ["Operating Income", "OperatingIncome"])
    tax_rate = 0.21  # approximate US corporate rate as a simplifying assumption
    if op_income_row is not None and equity is not None:
        nopat = float(op_income_row.iloc[-1]) * (1 - tax_rate)
        debt_val_tmp = float(total_debt_row.iloc[-1]) if total_debt_row is not None else 0
        cash_val_tmp = float(cash_row.iloc[-1]) if cash_row is not None else 0
        invested_capital = float(equity.iloc[-1]) + debt_val_tmp - cash_val_tmp
        roic = calc.safe_ratio(nopat, invested_capital)

    fcf_margin = None
    if fcf_row is not None and revenue is not None and len(revenue) > 0:
        fcf_margin = calc.safe_ratio(float(fcf_row.iloc[-1]), float(revenue.iloc[-1]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Gross Margin", fmt_pct(gross_margin))
    c2.metric("Operating Margin", fmt_pct(operating_margin))
    c3.metric("Net Margin", fmt_pct(net_margin))

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("FCF Margin", fmt_pct(fcf_margin))
    c5.metric("ROE", fmt_pct(roe))
    c6.metric("ROIC (approx.)", fmt_pct(roic))
    c7.metric("ROA", fmt_pct(roa))
    st.caption("ROIC is approximated using a flat 21% tax assumption on operating income — treat as directional.")

# ============================================================
# 5. FINANCIAL STRENGTH
# ============================================================
with tabs[4]:
    cash_val = float(cash_row.iloc[-1]) if cash_row is not None else None
    debt_val = float(total_debt_row.iloc[-1]) if total_debt_row is not None else None
    net_cash = (cash_val - debt_val) if (cash_val is not None and debt_val is not None) else None

    current_assets = get_row(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = get_row(balance, ["Current Liabilities", "Total Current Liabilities"])
    current_ratio = calc.safe_ratio(
        float(current_assets.iloc[-1]) if current_assets is not None else None,
        float(current_liabilities.iloc[-1]) if current_liabilities is not None else None,
    )

    interest_expense = get_row(income, ["Interest Expense"])
    interest_coverage = calc.safe_ratio(
        float(op_income_row.iloc[-1]) if op_income_row is not None else None,
        abs(float(interest_expense.iloc[-1])) if interest_expense is not None else None,
    )

    shares_now = info.get("sharesOutstanding")
    shares_row = get_row(balance, ["Share Issued", "Common Stock Shares Outstanding"])
    share_count_change_5y = calc.cagr_from_series(shares_row, 5) if shares_row is not None else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Cash", fmt_big(cash_val))
    c2.metric("Total Debt", fmt_big(debt_val))
    c3.metric("Net Cash", fmt_big(net_cash))

    c4, c5, c6 = st.columns(3)
    c4.metric("Debt / Equity", f"{info.get('debtToEquity') / 100:.2f}" if info.get("debtToEquity") else "N/A")
    c5.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A")
    c6.metric("Interest Coverage", f"{interest_coverage:.1f}x" if interest_coverage else "N/A")

    c7, c8 = st.columns(2)
    c7.metric("Shares Outstanding", fmt_big(shares_now) if shares_now else "N/A")
    c8.markdown(
        f"**Share Count Change (5Y, buybacks vs. dilution)**<br>{signed_pct_html(share_count_change_5y)}",
        unsafe_allow_html=True,
    )
    st.caption("Positive = share count grew (dilution). Negative = share count shrank (buybacks).")

# ============================================================
# 6. TECHNICALS (simple, by design)
# ============================================================
with tabs[5]:
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ma50_now = float(ma50.iloc[-1]) if len(ma50.dropna()) else None
    ma200_now = float(ma200.iloc[-1]) if len(ma200.dropna()) else None

    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    avg_vol = float(prices["Volume"].tail(20).mean())
    today_vol = float(prices["Volume"].iloc[-1])
    rel_strength = calc.relative_strength_vs_spy(close, spy_close) if spy_close is not None else None

    c1, c2 = st.columns(2)
    c1.metric("50-Day MA", fmt_money(ma50_now) if ma50_now else "N/A")
    c2.metric("200-Day MA", fmt_money(ma200_now) if ma200_now else "N/A")

    dist_ma50 = calc.distance_pct(current_price, ma50_now)
    dist_ma200 = calc.distance_pct(current_price, ma200_now)
    c3, c4 = st.columns(2)
    c3.markdown(f"**Distance from MA50**<br>{signed_pct_html(dist_ma50 / 100 if dist_ma50 is not None else None)}", unsafe_allow_html=True)
    c4.markdown(f"**Distance from MA200**<br>{signed_pct_html(dist_ma200 / 100 if dist_ma200 is not None else None)}", unsafe_allow_html=True)

    dist_high = calc.distance_pct(current_price, high_52w)
    dist_low = calc.distance_pct(current_price, low_52w)
    c5, c6 = st.columns(2)
    c5.markdown(f"**Distance from 52W High**<br>{signed_pct_html(dist_high / 100 if dist_high is not None else None)}", unsafe_allow_html=True)
    c6.markdown(f"**Distance from 52W Low**<br>{signed_pct_html(dist_low / 100 if dist_low is not None else None)}", unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    c7.markdown(f"**Relative Strength vs SPY (3mo)**<br>{signed_pct_html(rel_strength)}", unsafe_allow_html=True)
    c8.metric("Avg Daily Volume (20d)", f"{avg_vol:,.0f}")
    c9.markdown(f"**Today's Volume vs Avg**<br>{signed_pct_html((today_vol / avg_vol - 1) if avg_vol else None)}", unsafe_allow_html=True)

# ============================================================
# 7. OPTIONS DASHBOARD (highest priority — conservative, long-term framing)
# ============================================================
with tabs[6]:
    st.caption(
        "Conservative strike selection for long-term investors: targets a low, deliberate "
        "probability of assignment (~20% delta) rather than maximizing premium collected. "
        "Estimates are simplified Black-Scholes, not live market pricing — treat as planning "
        "guidance, not an execution price."
    )

    expirations = df_.get_option_expirations(ticker)
    if not expirations:
        st.warning(f"No options chain available for {ticker} on Yahoo Finance.")
    else:
        dtes = [(pd.to_datetime(e).date() - datetime.today().date()).days for e in expirations]
        target_dte = calc.pick_expiration_near_target(dtes, calc.TARGET_DTE_DAYS)
        target_expiration = expirations[dtes.index(target_dte)]

        calls_df, puts_df = df_.get_option_chain(ticker, target_expiration)

        if calls_df.empty and puts_df.empty:
            st.warning("Options chain returned no data for this expiration. Try again shortly.")
        else:
            atm_iv = calc.find_atm_iv(calls_df, puts_df, current_price)
            hv_30_opt = calc.historical_volatility(close, 30)
            iv_hv_ratio = calc.safe_ratio(atm_iv, hv_30_opt)
            exp_move = calc.expected_move(current_price, atm_iv, target_dte)

            # Log today's ATM IV snapshot so real IV Rank/Percentile builds up over time.
            storage.log_iv_snapshot(ticker, atm_iv)
            iv_history_rows = storage.get_iv_history(ticker)
            iv_history_values = [row[1] for row in iv_history_rows]

            if len(iv_history_values) >= 20:
                iv_rank, iv_percentile = calc.iv_rank_and_percentile(atm_iv, iv_history_values[:-1])
                iv_source_note = f"Based on {len(iv_history_values)} days of logged IV history for {ticker}."
            else:
                iv_rank, iv_percentile = calc.iv_rank_proxy_from_hv(close, atm_iv)
                iv_source_note = (
                    f"⚠️ Approximated using historical volatility (only {len(iv_history_values)}/20 days of real IV "
                    "history collected so far — check back daily and this will switch to true IV Rank automatically)."
                )

            st.markdown(f"**Expiration used for suggestions: {target_expiration} ({target_dte} DTE)**")

            c1, c2, c3 = st.columns(3)
            c1.metric("ATM Implied Volatility", fmt_pct(atm_iv))
            c2.metric("30-Day Historical Volatility", fmt_pct(hv_30_opt))
            c3.metric("IV / HV Ratio", f"{iv_hv_ratio:.2f}" if iv_hv_ratio else "N/A")

            c4, c5, c6 = st.columns(3)
            c4.metric("IV Rank", f"{iv_rank:.0f}" if iv_rank is not None else "N/A")
            c5.metric("IV Percentile", f"{iv_percentile:.0f}th" if iv_percentile is not None else "N/A")
            c6.metric("Expected Move (by expiration)", fmt_money(exp_move) if exp_move else "N/A")
            st.caption(iv_source_note)

            st.markdown("---")
            st.markdown("### Suggested Positions")

            csp = calc.pick_conservative_strike(puts_df, current_price, target_dte, option_type="put")
            cc = calc.pick_conservative_strike(calls_df, current_price, target_dte, option_type="call")

            col_csp, col_cc = st.columns(2)
            with col_csp:
                st.markdown("#### 🛡️ Cash Secured Put")
                if csp:
                    premium = calc.estimate_premium(csp)
                    ann_yield = calc.annualized_yield(premium, csp["strike"], target_dte)
                    prob_assignment = abs(csp["delta"]) * 100
                    st.metric("Suggested Strike", fmt_money(csp["strike"]))
                    st.metric("Suggested Delta", f"{csp['delta']:.2f}")
                    st.metric("Suggested DTE", f"{target_dte} days")
                    st.metric("Est. Premium (mid)", fmt_money(premium) if premium else "N/A")
                    st.metric("Estimated Annualized Yield", fmt_pct(ann_yield) if ann_yield else "N/A")
                    st.metric("Probability of Assignment", f"{prob_assignment:.0f}%")
                    st.metric("Probability OTM (expires worthless)", f"{100 - prob_assignment:.0f}%")
                else:
                    st.info("No suitable conservative put strike found in this chain.")

            with col_cc:
                st.markdown("#### 📈 Covered Call")
                if cc:
                    premium = calc.estimate_premium(cc)
                    ann_yield = calc.annualized_yield(premium, current_price, target_dte)
                    prob_assignment = cc["delta"] * 100
                    st.metric("Suggested Strike", fmt_money(cc["strike"]))
                    st.metric("Suggested Delta", f"{cc['delta']:.2f}")
                    st.metric("Suggested DTE", f"{target_dte} days")
                    st.metric("Est. Premium (mid)", fmt_money(premium) if premium else "N/A")
                    st.metric("Estimated Annualized Yield", fmt_pct(ann_yield) if ann_yield else "N/A")
                    st.metric("Probability of Assignment", f"{prob_assignment:.0f}%")
                    st.metric("Probability OTM (expires worthless)", f"{100 - prob_assignment:.0f}%")
                else:
                    st.info("No suitable conservative call strike found in this chain.")

            st.caption(
                f"Targets ~20% delta ({calc.TARGET_DELTA_CSP} for puts / {calc.TARGET_DELTA_CC} for calls) — "
                "a deliberately conservative, low-assignment-probability selection, not the richest premium available."
            )

# ============================================================
# 8. HISTORICAL PERFORMANCE
# ============================================================
with tabs[7]:
    cagr_10y = calc.cagr(close, 10)
    cagr_20y = calc.cagr(close, 20)
    mdd = calc.max_drawdown(close)
    best_year, best_ret, worst_year, worst_ret = calc.best_worst_calendar_year(close)
    roll_5y = calc.rolling_return(close, 5)
    roll_10y = calc.rolling_return(close, 10)
    beta = calc.beta_vs_benchmark(close, spy_close) if spy_close is not None else info.get("beta")
    hv_30 = calc.historical_volatility(close, 30)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**10-Year CAGR**<br>{signed_pct_html(cagr_10y)}", unsafe_allow_html=True)
    c2.markdown(f"**20-Year CAGR**<br>{signed_pct_html(cagr_20y)}", unsafe_allow_html=True)
    c3.markdown(f"**Max Drawdown**<br>{signed_pct_html(mdd)}", unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    c4.metric("Best Calendar Year", f"{best_year} ({best_ret*100:+.1f}%)" if best_year else "N/A")
    c5.metric("Worst Calendar Year", f"{worst_year} ({worst_ret*100:+.1f}%)" if worst_year else "N/A")

    c6, c7 = st.columns(2)
    c6.markdown(f"**Avg Rolling 5Y Return (annualized)**<br>{signed_pct_html(roll_5y)}", unsafe_allow_html=True)
    c7.markdown(f"**Avg Rolling 10Y Return (annualized)**<br>{signed_pct_html(roll_10y)}", unsafe_allow_html=True)

    c8, c9 = st.columns(2)
    c8.metric("Beta (vs SPY)", f"{beta:.2f}" if beta else "N/A")
    c9.metric("30-Day Historical Volatility", fmt_pct(hv_30))

# ============================================================
# 9. BUSINESS QUALITY SCORE
# ============================================================
with tabs[8]:
    metrics = {
        "roic": roic,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "pe_percentile": pe_percentile_5y,
        "peg_ratio": info.get("pegRatio"),
        "revenue_growth_yoy": rev_growth,
        "eps_growth_yoy": eps_growth,
        "debt_to_equity": (info.get("debtToEquity") / 100) if info.get("debtToEquity") else None,
        "current_ratio": current_ratio,
        "interest_coverage": interest_coverage,
        "historical_volatility_30d": hv_30,
    }
    category_scores, overall = calc.compute_quality_scores(metrics, sector=info.get("sector"))

    for category, score in category_scores.items():
        st.markdown(f"**{category}**  {calc.score_to_stars(score)}  ({score if score is not None else 'N/A'} / 5)")

    st.markdown("---")
    st.markdown(f"### Overall Score: {overall if overall is not None else 'N/A'} / 10")
    sector_note = (
        f"Thresholds adjusted for **{info.get('sector')}** as a capital-intensive sector "
        "(more debt / thinner margins expected and scored accordingly)."
        if info.get("sector") in calc.CAPITAL_INTENSIVE_SECTORS
        else f"Sector: **{info.get('sector', 'N/A')}** — standard thresholds applied."
    )
    st.caption(sector_note)
    st.caption(
        "Rule-based scoring, not AI-generated. Thresholds favor durable, moderately-valued, "
        "financially strong businesses over high-growth/high-volatility names — tune the "
        "constants in calculations.py if your philosophy differs."
    )

# ============================================================
# 10. INVESTMENT JOURNAL
# ============================================================
with tabs[9]:
    st.caption(
        "⚠️ Saved locally to a SQLite file. On Streamlit Community Cloud this can be wiped on "
        "redeploy — export a backup after adding notes, and re-import if you ever redeploy fresh."
    )
    entry = storage.get_entry(ticker)

    with st.form("journal_form"):
        reasons_owned = st.text_area("Reasons I Own This Company", value=entry.get("reasons_owned", ""))
        competitive_advantages = st.text_area("Competitive Advantages", value=entry.get("competitive_advantages", ""))
        long_term_thesis = st.text_area("Long-Term Thesis", value=entry.get("long_term_thesis", ""))
        biggest_risks = st.text_area("Biggest Risks", value=entry.get("biggest_risks", ""))
        fair_value_estimate = st.text_input("Fair Value Estimate", value=entry.get("fair_value_estimate", ""))
        ideal_buy_price = st.text_input("Ideal Buy Price", value=entry.get("ideal_buy_price", ""))
        ideal_csp_strike = st.text_input("Ideal CSP Strike", value=entry.get("ideal_csp_strike", ""))
        min_covered_call_strike = st.text_input("Minimum Covered Call Strike", value=entry.get("min_covered_call_strike", ""))
        sell_triggers = st.text_area("Things That Would Make Me Sell", value=entry.get("sell_triggers", ""))

        submitted = st.form_submit_button("💾 Save Journal Entry")
        if submitted:
            storage.save_entry(ticker, {
                "reasons_owned": reasons_owned,
                "competitive_advantages": competitive_advantages,
                "long_term_thesis": long_term_thesis,
                "biggest_risks": biggest_risks,
                "fair_value_estimate": fair_value_estimate,
                "ideal_buy_price": ideal_buy_price,
                "ideal_csp_strike": ideal_csp_strike,
                "min_covered_call_strike": min_covered_call_strike,
                "sell_triggers": sell_triggers,
            })
            st.success(f"Saved journal entry for {ticker}.")

    st.markdown("---")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "⬇️ Export Journal Backup (all tickers)",
            data=storage.export_backup_json(),
            file_name=f"journal_backup_{datetime.today().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
    with b2:
        uploaded = st.file_uploader("⬆️ Restore from Backup", type=["json"])
        if uploaded is not None:
            storage.import_backup_json(uploaded.read().decode("utf-8"))
            st.success("Journal restored from backup. Reload the page to see all entries.")
