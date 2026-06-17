# Small Cap Scanner — Trading System Methodology

*Updated March 20, 2026*

---

## Overview

This system trades both sides of earnings-driven MA crossover signals. **Bullish signals** with strong EPS and confirmed trend changes are traded long. **Bearish signals** — failed bullish setups from weaker configs or discretionary chart rejects — are traded short. The long side drives outsized returns; the short side adds consistent smaller gains and nearly doubles trade frequency.

---

## Part A: Long Strategy (Bullish Signals)

### Step 1: Scanner Signal (Quantitative)

The scanner detects a **20/50 SMA bullish crossover** occurring within N days of an earnings report that shows significant EPS growth. This combines a fundamental catalyst (earnings surprise) with a technical confirmation (trend change).

#### Signal Configs

Run both configs and deduplicate signals by (ticker, date):

| Config | EPS Threshold | Trend Window |
|--------|--------------|--------------|
| Tight  | >= 25%       | 30 days      |
| Broad  | >= 30%       | 60 days      |

#### Quantitative Filters (C17)

All of the following must be true:

1. **Direction**: Bullish only — 20-day SMA crosses above 50-day SMA
2. **EPS change**: Positive and above the config threshold
3. **Liquidity**: Average dollar volume >= $500K/day
4. **EPS magnitude**: |EPS change| < 100% (filters out penny stock distortions)
5. **Timing**: > 10 days between earnings report date and trend date (not a same-day reaction)
6. **Volume confirmation**: Signal day volume >= 1.2x the 20-day average volume — **no upper cap**

> **Why no volume cap?** The biggest winners all had extreme volume at signal: OPEN (2.7x), FCEL (3.2x), IBRX (2.4x). Capping at 2x would have excluded every major winner. High volume confirms institutional participation and conviction.

### Step 2: Discretionary Chart Review (Before Entry)

After the scanner flags a signal, **visually confirm the stock's price structure** before committing capital. The scanner finds the quantitative setup; your eyes confirm whether the stock is actually changing trend.

#### Do NOT Enter Long If:

**1. Unfilled Gaps**
Recent gaps near the current price act as magnets. The stock will chop until the gap fills rather than trending. Wait for the gap to fill before considering entry.
- *Example: NRDS had a gap on 5/7/25 at $10.20. Stock traded rangebound until the gap filled with another big gap lower on 8/8.*
- *Example: ARRY had two major gaps preceding the signal. Gap filled by 5/22 — that may have been a valid delayed entry.*

**2. No Trend Change**
The EPS report and MA crossover fired, but the stock is still trading in a sideways range or consolidation. The price structure hasn't actually changed. Identify the range's resistance (top) and support (bottom). Need a breakout above the range to confirm.
- *Example: U was sideways after EPS report. No breakout until May 28th — 15 days after the signal.*
- *Example: BILL was in a downtrend since Feb 2025. The mid-March swing high was resistance. No cross above it occurred.*
- *Example: PLUG was consolidating, not changing trend.*

**3. Still Trending Down**
The stock is in a downtrend and hasn't broken above a prior swing high or resistance level. The MA cross is occurring within a downtrend, not confirming a reversal.
- *Example: SPCE's low of 7/1 was support, but there was no trend change. A close above $3 would have confirmed a small reversal.*

#### DO Enter Long When:

Price structure confirms a new trend:
- Breakout above a consolidation range
- Close above a prior resistance/swing high
- Stock already trending up with the crossover confirming momentum

> **Impact of this filter:** In backtesting, the discretionary filter removed 6 of 7 losers without removing a single winner. Only XERS was a legitimate loss — the stock was already trending up and had closed above its prior highs. The stop-out was simply a normal loss on a valid trade.

#### Signals That Fail the Discretionary Review → Become Short Candidates

If a bullish signal from the current system (EPS>=25%/30d or EPS>=30%/60d) fails the chart review — unfilled gaps, no trend change, or still trending down — it becomes a **bearish signal** for the short strategy (see Part B).

### Step 3: Long Entry

- **When**: Buy in the **last 30 minutes of the trigger day** (Day 0 close)
- **Why D0 close**: You can see the full day's candle and volume before committing. D0 close is cheaper than D1 open 67% of the time (avg +4% savings). Earlier entry = better price, higher profit factor.

### Step 4: Long Stop Loss

| Day After Entry | Stop Rule |
|----------------|-----------|
| Day 1 | Exit if close < previous day's low |
| Day 2+ | Trailing stop: exit if close < highest close − 0.5× ATR(14) |

- **Max hold**: 15 days (though all trades resolve within ~8 days)
- **ATR period**: 14 days (10d and 20d tested — identical results)

#### Why 0.5x ATR Is Optimal

- **Winners** peak in an average of 3.7 days, then pull back. The stop captures the bulk of the move. 71% of winners drop further after the stop fires — confirming good exit timing.
- **Losers** exit in an average of 2.0 days. 60% of stopped-out losers kept falling — the stop is protecting capital.
- **Tighter (0.25-0.4x)**: Similar WR but flips marginal winners to losers.
- **Wider (0.75-1.5x)**: Lets losers bleed more and gives back gains on winners (stocks spike then mean-revert).

### Step 5: Long Exit

- **No profit targets** — capping winners destroys the system. The big winners (+93%, +129%) are what drive all the returns.
- Let the trailing stop do the work.
- **Do not re-enter** after being stopped out. Re-entry win rate is only 25% across all signals tested. Accept the loss and wait for the next fresh signal.

---

## Part B: Short Strategy (Bearish Signals)

### Signal Sources

Bearish signals come from two sources:

#### Source 1: Weaker Config Signals (EPS >= 20%)

Run an additional config pair and deduplicate by (ticker, date):

| Config | EPS Threshold | Trend Window |
|--------|--------------|--------------|
| Bear 1 | >= 20%       | 45 days      |
| Bear 2 | >= 20%       | 60 days      |

These signals must pass the same C17 quantitative filters (bullish crossover, EPS > 0, liquidity >= $500K, |EPS| < 100%, timing > 10 days, volume >= 1.2x). **Exclude any signal that already appears in the bullish system** (same ticker + date in the EPS>=25%/30d or EPS>=30%/60d configs).

> **Why these work as shorts:** The EPS growth (20-29%) isn't strong enough to drive a real trend change. The bullish crossover fires but lacks the fundamental conviction to sustain momentum, so the stock fades. Tighter EPS thresholds (25%+) don't produce enough unique short signals; looser thresholds (10%) add too many noisy trades.

#### Source 2: Discretionary Rejects

Any bullish signal from the current long system (EPS>=25%/30d or EPS>=30%/60d) that **fails the discretionary chart review** becomes a short candidate. These are stocks where:
- The quantitative signal fired (EPS + MA cross + volume)
- But the chart shows the trend hasn't actually changed (gaps, consolidation, continued downtrend)

> **Why these work as shorts:** Your chart reading is identifying "this bullish setup will fail." A failed bullish catalyst with no trend change means the stock is likely to continue its prior direction — down. In backtesting, discretionary rejects shorted at 75% WR, PF 6.11.

### Short Entry

- **When**: Short in the **last 30 minutes of the trigger day** (Day 0 close) — same timing as long entry
- Same rationale: see the full day's action before committing

### Short Stop Loss (Inverse of Long)

| Day After Entry | Cover Rule |
|----------------|-----------|
| Day 1 | Cover if close > previous day's high (breakout against you) |
| Day 2+ | Trailing cover: cover if close > lowest close + 0.5× ATR(14) |

- **Max hold**: 15 days
- **ATR period**: 14 days (same as long side)

### Short Exit

- Let the trailing cover stop do the work.
- No profit targets on shorts either — some short winners run +17-20%.

---

## Performance Summary

### Long Strategy — With Discretionary Filter (8 trades, May 2025 – Feb 2026)

| Metric | Value |
|--------|-------|
| **Win Rate** | **87.5%** |
| **Avg Return** | **+32.8%** |
| **Median Return** | **+10.8%** |
| **Profit Factor** | **80.42** |
| Avg Winner | +38.0% (5.1 day hold) |
| Avg Loser | −3.3% (2.0 day hold) |
| Avg Hold (all) | 4.8 days |
| Frequency | ~0.8 trades/month |
| Gross Gains | +266.1% |
| Gross Losses | −3.3% |

### Short Strategy — EPS>=20% Configs + Discretionary Rejects (8 trades)

| Metric | Value |
|--------|-------|
| **Win Rate** | **75.0%** |
| **Avg Return** | **+6.4%** |
| **Profit Factor** | **6.54** |
| Avg Winner | +10.0% (8.2 day hold) |
| Avg Loser | −4.6% (4.0 day hold) |
| Avg Hold (all) | 7.1 days |
| Frequency | ~0.8 trades/month |
| Gross Gains | +60.2% |
| Gross Losses | −9.2% |

### Combined System (16 trades)

| Metric | Value |
|--------|-------|
| **Win Rate** | **81.3%** |
| **Avg Return** | **+14.0%** (longs drive this) |
| **Profit Factor** | **12.27** |
| Avg Winner | +23.4% (6.5 day hold) |
| Avg Loser | −3.6% (3.3 day hold) |
| Avg Hold (all) | 5.9 days |
| Frequency | ~1.6 trades/month |
| Gross Gains | +326.3% |
| Gross Losses | −12.5% |

### Individual Long Trades (After Discretionary Filter)

| Ticker | Date | EPS% | Signal Vol | Entry$ | Exit$ | Return | Hold | W/L |
|--------|------|------|-----------|--------|-------|--------|------|-----|
| HIMS | 2025-05-09 | +65% | 1.4x | $51.96 | $61.12 | +17.6% | 3d | W |
| PUBM | 2025-05-09 | +39% | 2.6x | $11.09 | $11.42 | +3.0% | 5d | W |
| SPT | 2025-05-21 | +48% | 1.3x | $21.55 | $21.72 | +0.8% | 11d | W |
| XERS | 2025-07-14 | +67% | 1.8x | $5.44 | $5.26 | −3.3% | 2d | L |
| OPEN | 2025-07-16 | +36% | 2.7x | $1.49 | $2.88 | +93.3% | 4d | W |
| FCEL | 2025-09-17 | +37% | 3.2x | $7.65 | $9.08 | +18.7% | 4d | W |
| IBRX | 2026-01-13 | +38% | 2.4x | $2.82 | $6.45 | +128.7% | 7d | W |
| APPS | 2026-01-16 | +57% | 1.3x | $4.99 | $5.19 | +4.0% | 2d | W |

### Individual Short Trades

| Ticker | Date | Source | Signal Vol | Short$ | Cover$ | Return | Hold | W/L |
|--------|------|--------|-----------|--------|--------|--------|------|-----|
| CRMD | 2025-05-09 | EPS>=20% config | 2.2x | $11.66 | $12.28 | −5.3% | 5d | L |
| AI | 2025-05-12 | EPS>=20% config | 1.2x | $24.20 | $24.16 | +0.2% | 4d | W |
| NRDS | 2025-05-12 | Disc. reject | 1.3x | $11.85 | $10.75 | +9.3% | 12d | W |
| U | 2025-05-13 | Disc. reject | 1.2x | $21.95 | $22.81 | −3.9% | 3d | L |
| ARRY | 2025-05-14 | Disc. reject | 1.5x | $8.31 | $7.67 | +7.7% | 4d | W |
| LSPD | 2025-11-10 | EPS>=20% config | 1.5x | $13.32 | $11.11 | +16.6% | 11d | W |
| TIGR | 2026-01-12 | EPS>=20% config | 3.7x | $10.32 | $8.31 | +19.5% | 15d | W |
| PLUG | 2026-01-22 | Disc. reject | 2.0x | $2.59 | $2.41 | +6.9% | 3d | W |

---

## What NOT to Do

These were all tested and made results worse:

| Idea | Result | Why It Failed |
|------|--------|---------------|
| Profit targets | Destroyed returns | Big winners (+93%, +129%) drive the system |
| Wider stops for high-vol signals | −39pp on OPEN alone | Stocks spike then mean-revert; wider stop holds through the pullback |
| RSI > 80 exit override | Avg return dropped from +19.7% to +14.8% | Delays exit to a worse price |
| Re-entering stopped-out losers | 25% WR, +1.2% combined return | Too choppy and unreliable |
| Volume cap at 2x | Excluded every big winner | OPEN 2.7x, FCEL 3.2x, IBRX 2.4x |
| Oversold indicators (RSI, Bollinger, Stochastic) | 0 signals qualified | This is a momentum system — overbought is a feature, not a bug |
| Green candle requirement | Missed OPEN +94%, RDW +51% | Too many big winners have red signal-day candles |
| Two-day confirmation | WR dropped to 26.7% | Too strict, filters out winners |
| Shorting bullish system signals | 33.3% WR, PF 0.56 | Confirmed those signals are genuinely bullish — don't bet against them |
| EPS>=10%/60d as short config | 45.5% WR, PF 2.84 | Casts too wide a net; adds 7 noisy short trades that dilute the edge |

---

## Key Characteristics of This System

- **This is a momentum swing trade.** It captures the initial 3-7 day burst after an earnings-driven trend change (longs) or the fade when the catalyst isn't strong enough (shorts).
- **Returns are heavily skewed on the long side.** Most long winners are small (+1-18%), but occasional big winners (+93%, +129%) drive the majority of returns. You must take every long signal because you can't predict which will be the big one.
- **Short returns are more consistent.** Short winners average +10% in 8 days. No blowout winners, but reliable.
- **Losers are small and fast on both sides.** Long avg loss: −3.3% in 2 days. Short avg loss: −4.6% in 4 days.
- **The broader trend continues** after the long stop fires (avg 31 days to ultimate high, stocks often gain +100-300% more), but the pullback after exit is brutal (avg −30% drawdown from exit). Holding through or re-entering does not reliably capture the continuation.
- **Frequency doubles with shorts.** Long-only: ~0.8 trades/month. Combined: ~1.6 trades/month. Expanding the stock universe would increase signal count further.

---

## Data & Limitations

- **Sample size**: 16 trades (8 long + 8 short) over ~10 months. This is suggestive, not statistically definitive.
- **Single regime**: All signals appeared April 2025 onward. Not validated across multiple market cycles.
- **Discretionary element**: The chart review filter is subjective. Different traders may interpret price structure differently. The discretionary reject → short pipeline depends on this judgment.
- **Short-side borrowing**: Not all small caps are easy to borrow for shorting. Borrow costs and availability are not modeled.
- **Stock-only**: No options returns estimated yet. Given the avg long winner of +38% in 5 days, ATM 2-week calls would amplify long returns significantly. Puts on short signals would similarly amplify.

---

## Stock Universe

183 tickers across US small/mid caps ($1-$50), ADRs, biotech, and clean energy. See `core/providers/yfinance_provider.py` for the full list.
