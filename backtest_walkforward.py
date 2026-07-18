#!/usr/bin/env python3
"""
Walk-forward validation + afternoon session-cut test for the ICT NAS100 strategy.

Two questions this answers, honestly:
  1) Is the edge real or curve-fit?  The shipped levers (15M mitigation freshness,
     1M entry depth, session end) were tuned on the FULL 2022-2025 file, so the
     headline 47.7% / PF 1.73 is in-sample. Here we OPTIMISE those levers on a
     train window only, then measure the chosen config on a later window it never
     saw (anchored/expanding walk-forward). If out-of-sample PF stays > 1 and the
     optimiser keeps re-picking similar params, the edge survives contact with
     unseen data. If OOS collapses, it was fit.
  2) Does cutting the weak afternoon (14:00-16:00 ET) help OOS, or is that just
     another in-sample data-mine?  Session end (kzEnd) is folded INTO the grid so
     the walk-forward optimiser is free to choose it - a fair test.

Engine is imported verbatim from backtest_v2.py (the trusted independent port),
so entry/exit/stop logic is byte-for-byte the same; only the three swept levers
and the train/test date windows change. HTF arrays that don't depend on the swept
levers are precomputed ONCE; only the cheap 1M trade loop re-runs per config.
"""
import numpy as np
import pandas as pd
from backtest_v2 import (load, resample, bias_engine, flow_engine,
                         strict_pivots, confirmed_index, CSV)

# ---- fixed params (unchanged from the shipped indicator) ----
PIV = 3
MSS_MAX_AGE = 30
SWEEP_LEN = 20
REV_MAX_AGE = 60
SL_BUF = 2.0
RR = 2.0
MAX_RISK = 55.0
MIN_RISK = 20.0
KZ_START = 9 * 60 + 30            # 09:30 ET (NY cash open) - fixed
EOD_MIN = 16 * 60 + 59
NEUTRAL_4H_STRUCTURE = True
NEUTRAL_LONG_ONLY = True
SWEEP_ALL = True

# ---- swept levers (the walk-forward grid) ----
GRID_MIT = [5, 10, 20, 40]        # 15M FVG mitigation freshness (bars)
GRID_FRAC = [0.0, 0.5, 1.0]       # 1M FVG entry depth (0=edge tap, 1=full fill)
GRID_KZEND = [13 * 60, 14 * 60, 16 * 60]   # session end: 13:00 / 14:00 / 16:00 ET


def precompute():
    """Everything independent of the swept levers, computed once (midnight 4H anchor)."""
    df = load(CSV)
    d4 = resample(df, "4h", "start_day")
    d15 = resample(df, "15min", "start_day")
    b4, b4bT, b4bB, b4sT, b4sB = bias_engine(d4.h.values, d4.l.values, d4.c.values, PIV)
    f4, _, _ = flow_engine(d4.h.values, d4.l.values, d4.c.values, PIV)
    f15, a15b, a15s = flow_engine(d15.h.values, d15.l.values, d15.c.values, PIV)

    t = df.index.asi8
    i4 = confirmed_index(t, d4.index, pd.Timedelta(hours=4))
    i15 = confirmed_index(t, d15.index, pd.Timedelta(minutes=15))
    valid = (i4 >= 0) & (i15 >= 0)
    j4 = np.where(i4 >= 0, i4, 0)
    j15 = np.where(i15 >= 0, i15, 0)

    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    P = dict(
        df=df, o=o, h=h, l=l, c=c, n=len(df),
        mins=(df.index.hour * 60 + df.index.minute).to_numpy(),
        year=df.index.year.to_numpy(),
        phC=strict_pivots(h, l, PIV)[0], plC=strict_pivots(h, l, PIV)[1],
        prevLo=pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy(),
        prevHi=pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy(),
        valid=valid,
        rawBias=b4[j4], f4=f4[j4], bullT4=b4bT[j4], bearB4=b4sB[j4],
        flow=f15[j15], a15b=a15b[j15], a15s=a15s[j15],
    )
    return P


def simulate(P, MIT_MAX_AGE, ENTRY_FRAC, KZ_END):
    """The v2 1M trade loop, parameterised on the three swept levers."""
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
        biasLong = rb == 1 or (rb == 0 and NEUTRAL_4H_STRUCTURE and f4[i] == 1)
        biasShort = rb == -1 or (rb == 0 and NEUTRAL_4H_STRUCTURE and not NEUTRAL_LONG_ONLY and f4[i] == -1)
        bullT4, bearB4 = bullT4a[i], bearB4a[i]
        flow = flowA[i]
        mitL = 0 <= a15bA[i] <= MIT_MAX_AGE
        mitS = 0 <= a15sA[i] <= MIT_MAX_AGE

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
                sl = base - SL_BUF
                entry = trigL
                if MIN_RISK > 0: sl = min(sl, entry - MIN_RISK)
                risk = entry - sl
                if not (MAX_RISK > 0 and risk > MAX_RISK) and flat:
                    tp = entry + RR * risk
                    rec = {"dir": 1, "entry_time": P["df"].index[i], "entry": entry,
                           "sl": sl, "tp": tp, "risk": risk, "year": year[i],
                           "type": "continuation" if rb == 1 else "neutral"}
                    if l[i] <= sl:
                        trades.append({**rec, "result": "SL", "points": sl - entry, "r": -1.0})
                    else:
                        pos = rec
        elif setupShort and not np.isnan(eST) and not eSdone and inKZ and i > eSbar:
            trigS = eSB + ENTRY_FRAC * (eST - eSB)
            if h[i] >= trigS:
                eSdone = True
                base = max(eST, swingHi) if not np.isnan(swingHi) else eST
                sl = base + SL_BUF
                entry = trigS
                if MIN_RISK > 0: sl = max(sl, entry + MIN_RISK)
                risk = sl - entry
                if not (MAX_RISK > 0 and risk > MAX_RISK) and flat:
                    tp = entry - RR * risk
                    rec = {"dir": -1, "entry_time": P["df"].index[i], "entry": entry,
                           "sl": sl, "tp": tp, "risk": risk, "year": year[i],
                           "type": "continuation" if rb == -1 else "neutral"}
                    if h[i] >= sl:
                        trades.append({**rec, "result": "SL", "points": entry - sl, "r": -1.0})
                    else:
                        pos = rec

    if pos is not None:
        pts = (c[-1] - pos["entry"]) * pos["dir"]
        trades.append({**pos, "result": "EOD", "points": pts, "r": pts / pos["risk"]})
    return pd.DataFrame(trades)


def metrics(tr):
    if len(tr) == 0:
        return dict(n=0, win=0.0, totR=0.0, pf=0.0, avgR=0.0)
    r = tr.r.to_numpy()
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    return dict(n=len(tr), win=(r > 0).mean() * 100, totR=r.sum(),
                pf=(pos / neg if neg else float("inf")), avgR=r.mean())


def fmt(m):
    return (f"n={m['n']:<4} win%={m['win']:4.1f}  totR={m['totR']:+7.1f}  "
            f"PF={m['pf']:4.2f}  avgR={m['avgR']:+.3f}")


def main():
    print("Precomputing HTF engines (once)...")
    P = precompute()
    yr = P["year"]
    print(f"1M bars: {P['n']:,}   years present: {sorted(set(yr))}\n")

    # cache every grid config's full-history trades once, then slice by window
    print(f"Running {len(GRID_MIT)*len(GRID_FRAC)*len(GRID_KZEND)} configs over full history...")
    cache = {}
    for mit in GRID_MIT:
        for frac in GRID_FRAC:
            for kz in GRID_KZEND:
                cache[(mit, frac, kz)] = simulate(P, mit, frac, kz)
    print("done.\n")

    def window(tr, years):
        return tr[tr.year.isin(years)]

    SHIPPED = (5, 1.0, 16 * 60)   # the config the indicator actually ships with

    # ---------- anchored walk-forward ----------
    folds = [
        ("2023", "2024", [2023], [2024]),
        ("2023-2024", "2025", [2023, 2024], [2025]),
    ]
    print("=" * 92)
    print("ANCHORED WALK-FORWARD  (optimise levers on TRAIN totR, measure frozen config on unseen TEST)")
    print("  grid: mit in {5,10,20,40}  frac in {0,0.5,1.0}  session-end in {13:00,14:00,16:00}")
    print("=" * 92)
    for tr_lab, te_lab, tr_yrs, te_yrs in folds:
        best_cfg, best_tot = None, -1e9
        for cfg, tr in cache.items():
            m = metrics(window(tr, tr_yrs))
            if m["n"] >= 20 and m["totR"] > best_tot:
                best_tot, best_cfg = m["totR"], cfg
        tr_m = metrics(window(cache[best_cfg], tr_yrs))
        te_m = metrics(window(cache[best_cfg], te_yrs))
        ship_te = metrics(window(cache[SHIPPED], te_yrs))
        # in hindsight, the best config ON the test window (the "cheat" ceiling)
        cheat_cfg, cheat = None, -1e9
        for cfg, tr in cache.items():
            m = metrics(window(tr, te_yrs))
            if m["n"] >= 20 and m["totR"] > cheat:
                cheat, cheat_cfg = m["totR"], cfg
        mit, frac, kz = best_cfg
        print(f"\nTrain {tr_lab}  ->  Test {te_lab}")
        print(f"  optimiser picked: mit={mit}  frac={frac}  session-end={kz//60}:00")
        print(f"    train (in-sample) : {fmt(tr_m)}")
        print(f"    TEST  (out-sample): {fmt(te_m)}   <-- the honest number")
        print(f"    shipped cfg (5/1.0/16:00) on same TEST: {fmt(ship_te)}")
        print(f"    best-on-test in hindsight {cheat_cfg}: totR={cheat:+.1f}  "
              f"(gap = optimiser left {cheat - te_m['totR']:+.1f}R on the table)")

    # ---------- afternoon session-cut, full sample (fixed shipped levers) ----------
    print("\n" + "=" * 92)
    print("AFTERNOON SESSION CUT  (shipped levers mit=5 frac=1.0; vary only session end), full sample")
    print("=" * 92)
    for kz in GRID_KZEND:
        tr = cache[(5, 1.0, kz)]
        allm = metrics(tr)
        print(f"\n09:30-{kz//60}:00 ET   ALL: {fmt(allm)}")
        for y in [2023, 2024, 2025]:
            print(f"      {y}: {fmt(metrics(window(tr, [y])))}")

    # ---------- what the shipped session (16:00) throws away ----------
    full = cache[(5, 1.0, 16 * 60)]
    cut = cache[(5, 1.0, 13 * 60)]
    dropped = full[~full.entry_time.isin(cut.entry_time)]
    print("\n" + "-" * 92)
    print(f"Trades in 13:00-16:00 window that the cut removes: {fmt(metrics(dropped))}")
    print("-" * 92)


if __name__ == "__main__":
    main()
