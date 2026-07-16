#!/usr/bin/env python3
"""
Missed-trade diagnosis for the ICT NQ system.

Minute-level census inside every kill zone: when the 15M flow gives a
direction, check each gate independently and tally the minutes where
exactly ONE gate blocks an otherwise fully-aligned setup. That gate is
"the reason" a trade was missed at that moment.

Also: day-level report of big-move kill zones (>=80 pts 09:15->12:00
drift) where the current system took zero trades.
"""
import numpy as np
import pandas as pd
from backtest_ict import (load_data, resample, bias_engine, flow_engine,
                          PIV_BIAS, PIV_FLOW, PIV_CHART, MSS_MAX_AGE,
                          MIT_MAX_AGE, SWEEP_LEN, REV_MAX_AGE,
                          KZ_START, KZ_END, SL_BUFFER, MAX_RISK)

df = load_data()
df4 = resample(df, "4h"); df15 = resample(df, "15min")
h4bias, h4bT, h4bB, h4sT, h4sB = bias_engine(
    df4["open"].to_numpy(), df4["high"].to_numpy(),
    df4["low"].to_numpy(), df4["close"].to_numpy(), PIV_BIAS)
m15flow, m15bAge, m15sAge = flow_engine(
    df15["open"].to_numpy(), df15["high"].to_numpy(),
    df15["low"].to_numpy(), df15["close"].to_numpy(), PIV_FLOW)
h4flow, _x, _y = flow_engine(
    df4["open"].to_numpy(), df4["high"].to_numpy(),
    df4["low"].to_numpy(), df4["close"].to_numpy(), PIV_BIAS)
t1 = df.index.asi8
idx4 = np.searchsorted((df4.index + pd.Timedelta(hours=4)).asi8, t1, "right") - 1
idx15 = np.searchsorted((df15.index + pd.Timedelta(minutes=15)).asi8, t1, "right") - 1

o = df["open"].to_numpy(); h = df["high"].to_numpy()
l = df["low"].to_numpy(); c = df["close"].to_numpy()
n = len(df)
mins = np.asarray(df.index.hour * 60 + df.index.minute)
day_ids = df.index.normalize()
prevLo = pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy()
prevHi = pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy()

lastPh = lastPl = np.nan
lastSwingHi = lastSwingLo = np.nan
mssDir = 0; mssBar = -10**9
sellSweepBar = buySweepBar = -10**9
# loose candidate FVGs (ungated) per direction
bTop = bBot = np.nan; bBar = -1; bDone = False
sTop = sBot = np.nan; sBar = -1; sDone = False

from collections import Counter
tally = Counter()
kz_minutes = 0

for i in range(n):
    if i >= 2 * PIV_CHART:
        p = i - PIV_CHART
        seg = h[i - 2 * PIV_CHART:i + 1]
        if h[p] == seg.max() and (seg == h[p]).sum() == 1:
            lastPh = h[p]; lastSwingHi = h[p]
        seg = l[i - 2 * PIV_CHART:i + 1]
        if l[p] == seg.min() and (seg == l[p]).sum() == 1:
            lastPl = l[p]; lastSwingLo = l[p]
    if not np.isnan(lastPh) and c[i] > lastPh:
        mssDir = 1; mssBar = i; lastPh = np.nan
    if not np.isnan(lastPl) and c[i] < lastPl:
        mssDir = -1; mssBar = i; lastPl = np.nan
    if not np.isnan(prevLo[i]):
        if l[i] < prevLo[i] and c[i] > prevLo[i]:
            sellSweepBar = i
        if h[i] > prevHi[i] and c[i] < prevHi[i]:
            buySweepBar = i
    # ungated candidate entry FVGs
    if i >= 2:
        if l[i] > h[i - 2]:
            bTop, bBot, bBar, bDone = l[i], h[i - 2], i, False
        if h[i] < l[i - 2]:
            sTop, sBot, sBar, sDone = l[i - 2], h[i], i, False
    if not np.isnan(bTop) and c[i] < bBot:
        bTop = bBot = np.nan
    if not np.isnan(sTop) and c[i] > sTop:
        sTop = sBot = np.nan

    if not (KZ_START <= mins[i] < KZ_END):
        continue
    j4 = idx4[i]; j15 = idx15[i]
    if j4 < 0 or j15 < 0:
        continue
    kz_minutes += 1
    flow = m15flow[j15]
    if flow == 0:
        tally["15M flow flat"] += 1
        continue
    bias = h4bias[j4]
    if bias == 0:
        bias = h4flow[j4]  # structure fallback (current config)
    g = {}
    g["4H bias"] = bias == flow
    g["15M FVG mitigated"] = (0 <= m15bAge[j15] <= MIT_MAX_AGE) if flow == 1 else (0 <= m15sAge[j15] <= MIT_MAX_AGE)
    g["1M MSS"] = mssDir == flow and (i - mssBar) <= MSS_MAX_AGE
    g["liquidity sweep"] = (i - sellSweepBar) <= REV_MAX_AGE if flow == 1 else (i - buySweepBar) <= REV_MAX_AGE
    if flow == 1:
        okf = not np.isnan(bTop) and not bDone
        if okf:
            base = min(bBot, lastSwingLo) if not np.isnan(lastSwingLo) else bBot
            okf = (bTop - (base - SL_BUFFER)) <= MAX_RISK
        g["1M entry FVG"] = okf
    else:
        okf = not np.isnan(sTop) and not sDone
        if okf:
            base = max(sTop, lastSwingHi) if not np.isnan(lastSwingHi) else sTop
            okf = ((base + SL_BUFFER) - sBot) <= MAX_RISK
        g["1M entry FVG"] = okf
    fails = [k for k, v in g.items() if not v]
    if len(fails) == 0:
        tally["ALL GATES PASS"] += 1
    elif len(fails) == 1:
        tally["blocked only by: " + fails[0]] += 1
    else:
        tally["2+ gates failing"] += 1

print(f"Kill-zone minutes analysed: {kz_minutes:,}\n")
for k, v in tally.most_common():
    print(f"{k:<38} {v:>7,}  ({v/kz_minutes*100:4.1f}%)")

# ---- day-level: big-move kill zones with no trade ----
tr = pd.read_csv("/Users/manjunathb/Documents/FAB/ICT Indicator/backtest_trades.csv",
                 parse_dates=["entry_time"])
traded_days = set(tr["entry_time"].dt.normalize())
kz_mask = (mins >= KZ_START) & (mins < KZ_END)
kzdf = pd.DataFrame({"day": day_ids[kz_mask], "o": o[kz_mask], "c": c[kz_mask]})
g = kzdf.groupby("day").agg(first=("o", "first"), last=("c", "last"))
g["move"] = (g["last"] - g["first"]).abs()
big = g[g["move"] >= 80]
missed = big[~big.index.isin(traded_days)]
print(f"\nKZ days total: {len(g)}   days traded: {len(set(traded_days) & set(g.index))}")
print(f"Big-move KZ days (>=80 pts drift): {len(big)}   of which NO trade: {len(missed)} "
      f"({len(missed)/len(big)*100:.0f}%)")
print("Sample missed big days:", [d.strftime('%Y-%m-%d') for d in list(missed.index)[-8:]])
