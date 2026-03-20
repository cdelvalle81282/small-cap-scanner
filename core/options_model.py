"""Black-Scholes ATM call option return estimator."""

import math

from scipy.stats import norm


def bs_call_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Standard Black-Scholes call price.

    Args:
        S: Current stock price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate (annualized).
        sigma: Implied volatility (annualized).

    Returns:
        Theoretical call option price.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def estimate_call_return(
    stock_return_pct: float,
    stock_price: float,
    iv: float = 0.60,
    dte: int = 14,
    hold_days: int = 5,
    r: float = 0.05,
) -> float:
    """Estimate ATM call return given a stock move.

    Args:
        stock_return_pct: Stock return in percent (e.g. 10.0 for +10%).
        stock_price: Stock price at entry.
        iv: Implied volatility (annualized, e.g. 0.60 for 60%).
        dte: Days to expiration at entry.
        hold_days: Number of days the option is held.
        r: Risk-free rate.

    Returns:
        Estimated call option return in percent.
    """
    if stock_price <= 0 or dte <= 0:
        return 0.0

    K = stock_price  # ATM
    T_entry = dte / 365.0
    entry_price = bs_call_price(stock_price, K, T_entry, r, iv)
    if entry_price <= 0:
        return 0.0

    new_stock = stock_price * (1 + stock_return_pct / 100.0)
    T_exit = max((dte - hold_days) / 365.0, 0.001)
    exit_price = bs_call_price(new_stock, K, T_exit, r, iv)

    return (exit_price - entry_price) / entry_price * 100.0
