#!/usr/bin/env python3
"""
Trend-regime guard test for the ICT NAS100 strategy (NQ 1-min, 09:30-13:00).

The neutral-4H-bias fallback currently leans LONG only (neutralLongOnly=true),
a choice fit to a 2022-2025 uptrend. This tests replacing that static rule with a
regime-following one: when 4H bias is neutral and we fall back to 4H structure,
take the fallback LONG only above the daily 200-SMA and SHORT only below it.

Purpose is insurance, not optimisation. The whole sample is a bull market, so the
guard should barely change the numbers here (regime is 'up' almost always) - that
is the point: it must be ~free on the data we have while protecting the one regime
the data does NOT contain (a sustained downtrend, where leaning long is the failure
mode). It CANNOT be validated for the downside case on this file - there is no bear
market in it - so it is justified by mechanism plus 'costs nothing in-sample'.

Explicit 4H FVG-bias trades (rawBias == +/-1) are NEVER gated by the guard; only
the discretionary neutral fallback is. Regime uses the last CONFIRMED daily bar
(no lookahead). SMA warmup (< min_periods days) defaults to 'up' = current behaviour.
"""
import numpy as np
import pandas as pd
from backtest_walkforward import (precompute, MSS_MAX_AGE, REV_MAX_AGE, SL_BUF,
                                  MAX_RISK, MIN_RISK, KZ_START, EOD_MIN,
                                  NEUTRAL_4H_STRUCTURE)
from backtest_v2 import resample, confirmed_index

MIT_MAX_AGE = 5
ENTRY_FRAC = 1.0
KZ_END = 13 * 60
RR = 2.0
SMA_LEN = 200          # daily SMA that defines the regime
SMA_WARMUP = 100       # days before which regime defaults to 'up'


def daily_regime(P):
    """Per-1m-bar boolean: is yesterday's daily close >= its 200-day SMA?"""
    df = P["df"]
    dD = resample(df, "1D", "start_day")
    sma = dD.c.rolling(SMA_LEN, min_periods=SMA_WARMUP).mean()
    up_daily = (dD.c.to_numpy() >= sma.to_numpy())
    up_daily = np.where(np.isnan(sma.to_numpy()), True, up_daily)   # warmup -> up
    iD = confirmed_index(df.index.asi8, dD.index, pd.Timedelta(days=1))
    regimeUp = np.where(iD >= 0, up_daily[np.clip(iD, 0, len(up_daily) - 1)], True)
    frac_down = 1.0 - up_daily.mean()
    return regimeUp, frac_down, len(dD)


def simulate_regime(P, regimeUp, guard):
    o, h, l, c, n = P["o"], P["h"], P["l"], P["c"], P["n"]
    mins, year, valid = P["mins"], P["year"], P["valid"]
    phC, plC, prevLo, prevHi = P["phC"], P["plC"], P["prevLo"], P["prevHi"]
    rawBias, f4, bullT4a, bearB4a = P["rawBias"], P["f4"], P["bullT4"], P["bearB4"]
    flowA, a15bA, a15sA = P["flow"], P["a15b"], P["a15s"]

    lastPh = lastPl = swingHi = swingLo = np.nan
    mssDir, mssBar = 0, -10**9
    sellSweepBar = buySweepBar = -10**9
    eBT = eBB = np.nan; eBbar = -1; eBdone = False
    eST = eSB = np.nan; eSbar = -1; eSdone = False
    pos = None
    trades = []

    for i in range(n):
        if pos is not None:
            px = res = None
            if pos["dir"] == 1:
                if l[i] <= pos["sl"]: px, res = pos["sl"], "SL"
                elif h[i] >= pos["tp"]: px, res = pos["tp"], "TP"
            else:
                if h[i] >= pos["sl"]: px, res = pos["sl"], "SL"
                elif l[i] <= pos["tp"]: px, res = pos["tp"], "TP"
            if px is None and mins[i] >= EOD_MIN:
                px, res = c[i], "EOD"
            if px is not None:
                pts = (px - pos["entry"]) * pos["dir"]
                trades.append({**pos, "result": res, "points": pts, "r": pts / pos["risk"]})
                pos = None

        if not np.isnan(phC[i]): lastPh = swingHi = phC[i]
        if not np.isnan(plC[i]): lastPl = swingLo = plC[i]
        if not np.isnan(lastPh) and c[i] > lastPh:
            mssDir, mssBar, lastPh = 1, i, np.nan
        if not np.isnan(lastPl) and c[i] < lastPl:
            mssDir, mssBar, lastPl = -1, i, np.nan
        mssRecent = (i - mssBar) <= MSS_MAX_AGE
        if not np.isnan(prevLo[i]):
            if l[i] < prevLo[i] and c[i] > prevLo[i]: sellSweepBar = i
            if h[i] > prevHi[i] and c[i] < prevHi[i]: buySweepBar = i

        if not valid[i]:
            continue
        rb = rawBias[i]
        up = regimeUp[i]
        # explicit 4H FVG bias is never gated; only the neutral fallback follows regime
        neutLong = rb == 0 and NEUTRAL_4H_STRUCTURE and f4[i] == 1 and (not guard or up)
        neutShort = rb == 0 and NEUTRAL_4H_STRUCTURE and f4[i] == -1 and (guard and not up)
        biasLong = rb == 1 or neutLong
        biasShort = rb == -1 or neutShort
        bullT4, bearB4 = bullT4a[i], bearB4a[i]
        flow, mitL, mitS = flowA[i], 0 <= a15bA[i] <= MIT_MAX_AGE, 0 <= a15sA[i] <= MIT_MAX_AGE

        setupLong = (biasLong and flow == 1 and mitL and mssDir == 1 and mssRecent
                     and (i - sellSweepBar) <= REV_MAX_AGE)
        setupShort = (biasShort and flow == -1 and mitS and mssDir == -1 and mssRecent
                      and (i - buySweepBar) <= REV_MAX_AGE)

        if i >= 2:
            if l[i] > h[i - 2] and setupLong:
                eBT, eBB, eBbar, eBdone = l[i], h[i - 2], i, False
            if h[i] < l[i - 2] and setupShort:
                eST, eSB, eSbar, eSdone = l[i - 2], h[i], i, False
        if not np.isnan(eBT) and c[i] < eBB: eBT = eBB = np.nan
        if not np.isnan(eST) and c[i] > eST: eST = eSB = np.nan

        inKZ = KZ_START <= mins[i] < KZ_END
        flat = pos is None

        if setupLong and not np.isnan(eBT) and not eBdone and inKZ and i > eBbar:
            trigL = eBT - ENTRY_FRAC * (eBT - eBB)
            if l[i] <= trigL:
                eBdone = True
                base = min(eBB, swingLo) if not np.isnan(swingLo) else eBB
                sl = base - SL_BUF; entry = trigL
                if MIN_RISK > 0: sl = min(sl, entry - MIN_RISK)
                risk = entry - sl
                ttype = "continuation" if rb == 1 else "neutral-long"
                if not (MAX_RISK > 0 and risk > MAX_RISK) and flat:
                    tp = entry + RR * risk
                    rec = {"dir": 1, "entry_time": P["df"].index[i], "entry": entry,
                           "sl": sl, "tp": tp, "risk": risk, "year": year[i], "type": ttype}
                    if l[i] <= sl:
                        trades.append({**rec, "result": "SL", "points": sl - entry, "r": -1.0})
                    else:
                        pos = rec
        elif setupShort and not np.isnan(eST) and not eSdone and inKZ and i > eSbar:
            trigS = eSB + ENTRY_FRAC * (eST - eSB)
            if h[i] >= trigS:
                eSdone = True
                base = max(eST, swingHi) if not np.isnan(swingHi) else eST
                sl = base + SL_BUF; entry = trigS
                if MIN_RISK > 0: sl = max(sl, entry + MIN_RISK)
                risk = sl - entry
                ttype = "continuation" if rb == -1 else "neutral-short"
                if not (MAX_RISK > 0 and risk > MAX_RISK) and flat:
                    tp = entry - RR * risk
                    rec = {"dir": -1, "entry_time": P["df"].index[i], "entry": entry,
                           "sl": sl, "tp": tp, "risk": risk, "year": year[i], "type": ttype}
                    if h[i] >= sl:
                        trades.append({**rec, "result": "SL", "points": entry - sl, "r": -1.0})
                    else:
                        pos = rec

    if pos is not None:
        pts = (c[-1] - pos["entry"]) * pos["dir"]
        trades.append({**pos, "result": "EOD", "points": pts, "r": pts / pos["risk"]})
    return pd.DataFrame(trades)


def stats(tr, label):
    r = tr.r.to_numpy()
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
    pos = r[r > 0].sum(); neg = -r[r < 0].sum(); pf = pos / neg if neg else float("inf")
    print(f"{label:<22} n={len(tr):<4} win%={(r>0).mean()*100:4.1f}  totR={r.sum():+7.1f}  "
          f"avgR={r.mean():+.3f}  PF={pf:4.2f}  maxDD={dd:+6.1f}R")


def main():
    print("Precomputing HTF engines (once)...")
    P = precompute()
    regimeUp, frac_down, ndays = daily_regime(P)
    print(f"Daily bars: {ndays}   regime 'down' (below {SMA_LEN}-day SMA): "
          f"{frac_down*100:.1f}% of days\n")

    off = simulate_regime(P, regimeUp, guard=False)   # == shipped behaviour
    on = simulate_regime(P, regimeUp, guard=True)

    print("=" * 84)
    print("SHIPPED (neutral fallback = longs only, static)")
    stats(off, "  ALL")
    for y in [2023, 2024, 2025]:
        stats(off[off.year == y], f"    {y}")
    print("-" * 84)
    print("REGIME GUARD (neutral fallback follows daily 200-SMA: long above / short below)")
    stats(on, "  ALL")
    for y in [2023, 2024, 2025]:
        stats(on[on.year == y], f"    {y}")
    print("=" * 84)

    def key(t):
        return set(zip(t.entry_time.astype("int64"), t.dir))
    ko, kon = key(off), key(on)
    removed = off[[k not in kon for k in zip(off.entry_time.astype("int64"), off.dir)]]
    added = on[[k not in ko for k in zip(on.entry_time.astype("int64"), on.dir)]]
    print(f"\nWhat the guard changed on this (bull) sample:")
    print(f"  neutral-long trades REMOVED (regime was down): {len(removed)}"
          f"  -> {removed.r.sum():+.1f}R" if len(removed) else
          "  neutral-long trades REMOVED: 0")
    print(f"  neutral-short trades ADDED  (regime was down): {len(added)}"
          f"  -> {added.r.sum():+.1f}R" if len(added) else
          "  neutral-short trades ADDED: 0")
    print(f"  net effect: {on.r.sum() - off.r.sum():+.1f}R  "
          f"({len(on)-len(off):+d} trades)")
    print("\nType mix (guard on):", on.type.value_counts().to_dict())


if __name__ == "__main__":
    main()
