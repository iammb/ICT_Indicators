#!/usr/bin/env python3
"""
Volatility-scaled stops: fixed points vs % of price, on both datasets.

The 2008-2020 OANDA run showed 56% of trades force-closing at EOD, because the
strategy's stop floor/cap are FIXED POINTS (20 / 55 pt) calibrated to the NQ
future at ~15-20k, but the cash index there traded at 2-8k - so 20-55 pt was a
1-3% move that rarely completed intraday. That is a price-SCALE artifact of the
proxy, not the edge failing.

This replaces the fixed point floor/cap/buffer with a PERCENT of entry price,
anchored so it reproduces the current 20 / 55 / 2 pt values at NQ's median price
(so it is neutral-by-construction on the tuning file) and auto-scales down at the
low cash-index levels. If the edge is real, the % version should:
  * stay ~unchanged on NQ 2022-2025 (anchor point), and
  * on OANDA 2008-2020, collapse the EOD force-closes and lift avgR/PF toward the
    strategy's true expectancy.

Everything else (entry logic, morning session, regime guard, 2R) is unchanged.
"""
import numpy as np
import pandas as pd
from backtest_v2 import load as load_nq, CSV
from backtest_10yr import load_oanda, build_P, DATA_DIR
from backtest_regime import daily_regime
from backtest_walkforward import (MSS_MAX_AGE, REV_MAX_AGE, KZ_START,
                                  EOD_MIN, NEUTRAL_4H_STRUCTURE)

MIT_MAX_AGE = 5
ENTRY_FRAC = 1.0
KZ_END = 13 * 60
RR = 2.0
# fixed-point defaults (current shipped)
FIX_MIN, FIX_MAX, FIX_BUF = 20.0, 55.0, 2.0


def simulate_vol(P, regimeUp, minr_p, maxr_p, buf_p, pct):
    """pct=False -> minr_p/maxr_p/buf_p are POINTS; pct=True -> fractions of entry price.
    Regime guard is always ON (shipped default)."""
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
        rb = rawBias[i]; up = regimeUp[i]
        neutLong = rb == 0 and NEUTRAL_4H_STRUCTURE and f4[i] == 1 and up
        neutShort = rb == 0 and NEUTRAL_4H_STRUCTURE and f4[i] == -1 and not up
        biasLong = rb == 1 or neutLong
        biasShort = rb == -1 or neutShort
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
                entry = trigL
                buf = buf_p * entry if pct else buf_p
                minr = minr_p * entry if pct else minr_p
                maxr = maxr_p * entry if pct else maxr_p
                base = min(eBB, swingLo) if not np.isnan(swingLo) else eBB
                sl = base - buf
                if minr > 0: sl = min(sl, entry - minr)
                risk = entry - sl
                ttype = "continuation" if rb == 1 else ("neutral-long" if neutLong else "other")
                if not (maxr > 0 and risk > maxr) and flat:
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
                entry = trigS
                buf = buf_p * entry if pct else buf_p
                minr = minr_p * entry if pct else minr_p
                maxr = maxr_p * entry if pct else maxr_p
                base = max(eST, swingHi) if not np.isnan(swingHi) else eST
                sl = base + buf
                if minr > 0: sl = max(sl, entry + minr)
                risk = sl - entry
                ttype = "continuation" if rb == -1 else ("neutral-short" if neutShort else "other")
                if not (maxr > 0 and risk > maxr) and flat:
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
    if len(tr) == 0:
        print(f"{label:<26} no trades"); return
    r = tr.r.to_numpy()
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
    pos = r[r > 0].sum(); neg = -r[r < 0].sum(); pf = pos / neg if neg else float("inf")
    mix = tr.result.value_counts().to_dict()
    print(f"{label:<26} n={len(tr):<5} win%={(r>0).mean()*100:4.1f}  totR={r.sum():+7.1f}  "
          f"avgR={r.mean():+.3f}  PF={pf:4.2f}  DD={dd:+6.1f}R  TP/SL/EOD="
          f"{mix.get('TP',0)}/{mix.get('SL',0)}/{mix.get('EOD',0)}")


def main():
    print("Loading NQ 2022-2025 (tuning file)...")
    nq = load_nq(CSV); Pnq = build_P(nq); rnq, _, _ = daily_regime(Pnq)
    ref = float(np.median(nq.c.values))
    MIN_PCT, MAX_PCT, BUF_PCT = FIX_MIN / ref, FIX_MAX / ref, FIX_BUF / ref
    print(f"NQ median price {ref:,.0f}  ->  % anchors: min {MIN_PCT*100:.3f}%  "
          f"max {MAX_PCT*100:.3f}%  buf {BUF_PCT*100:.4f}%\n")

    print("Loading OANDA NAS100 2008-2020 (unseen)...")
    oa = load_oanda(DATA_DIR); Poa = build_P(oa); roa, _, _ = daily_regime(Poa)
    print()

    print("=" * 104)
    print("NQ 2022-2025  (must stay ~unchanged - the % anchor is calibrated here)")
    print("=" * 104)
    stats(simulate_vol(Pnq, rnq, FIX_MIN, FIX_MAX, FIX_BUF, pct=False), "fixed points (20/55/2)")
    stats(simulate_vol(Pnq, rnq, MIN_PCT, MAX_PCT, BUF_PCT, pct=True), "% of price (scaled)")

    print("\n" + "=" * 104)
    print("OANDA NAS100 2008-2020  (the real out-of-sample read)")
    print("=" * 104)
    fx = simulate_vol(Poa, roa, FIX_MIN, FIX_MAX, FIX_BUF, pct=False)
    pc = simulate_vol(Poa, roa, MIN_PCT, MAX_PCT, BUF_PCT, pct=True)
    stats(fx, "fixed points (20/55/2)")
    stats(pc, "% of price (scaled)")
    print("\n  % of price, by year:")
    for y in sorted(pc.year.unique()):
        stats(pc[pc.year == y], f"    {y}")
    print("  net of costs (% version, pts/round-trip):", end=" ")
    for cost in (0.0, 1.0, 2.0):
        print(f"[{cost:.0f} -> {(pc.points-cost).div(pc.risk).sum():+.0f}R]", end=" ")
    print()


if __name__ == "__main__":
    main()
