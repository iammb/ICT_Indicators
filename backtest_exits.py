#!/usr/bin/env python3
"""
Exit-management test for the ICT NAS100 strategy (NQ 1-min, 09:30-13:00 session).

The shipped model has ZERO trade management: every trade is a binary +2R (target)
or -1R (stop). This asks whether managing the winners/losers after entry improves
the edge, holding the ENTRY logic fixed (shipped levers: mit=5, frac=1.0, morning
session). Exit policy changes how long a position is held, which changes which
later setups are taken (one position at a time), so every mode re-runs the full
1M loop rather than replaying trades in isolation.

Fill conventions (same conservative spirit as backtest_v2):
  * stop is always checked before any target, using the stop level in force at the
    START of the bar - a stop moved up to breakeven this bar can only trigger on a
    LATER bar (never same-bar as the +1R that armed it)
  * the +1R arm/scale/trail is detected from the bar's favourable extreme, and only
    affects subsequent bars
  * points and R are size-weighted; a half scaled off at +1R banks 0.5R
  * anything still open at 12:59 ET is force-closed at that bar's close

Modes:
  baseline    fixed 2R target, -1R stop (the shipped behaviour)
  be_1r       2R target; at +1R move stop to breakeven
  scale_1r    2R target; at +1R sell half (bank 0.5R) and move stop to breakeven
  trail_1r    no fixed target; at +1R start trailing the stop 1R behind the extreme
  target_3r   fixed 3R target, -1R stop (let winners run further, no BE)
  be_1r_3r    3R target; at +1R move stop to breakeven
"""
import numpy as np
import pandas as pd
from backtest_walkforward import (precompute, PIV, MSS_MAX_AGE, REV_MAX_AGE,
                                  SL_BUF, MAX_RISK, MIN_RISK, KZ_START, EOD_MIN,
                                  NEUTRAL_4H_STRUCTURE, NEUTRAL_LONG_ONLY)

MIT_MAX_AGE = 5
ENTRY_FRAC = 1.0
KZ_END = 13 * 60
RR = 2.0          # base target for target-based modes (overridden to 3 where noted)


def simulate_exits(P, mode):
    o, h, l, c, n = P["o"], P["h"], P["l"], P["c"], P["n"]
    mins, year, valid = P["mins"], P["year"], P["valid"]
    phC, plC, prevLo, prevHi = P["phC"], P["plC"], P["prevLo"], P["prevHi"]
    rawBias, f4, bullT4a, bearB4a = P["rawBias"], P["f4"], P["bullT4"], P["bearB4"]
    flowA, a15bA, a15sA = P["flow"], P["a15b"], P["a15s"]
    tgt = 3.0 if mode in ("target_3r", "be_1r_3r") else RR
    use_be = mode in ("be_1r", "be_1r_3r")
    use_scale = mode == "scale_1r"
    use_trail = mode == "trail_1r"

    lastPh = lastPl = swingHi = swingLo = np.nan
    mssDir, mssBar = 0, -10**9
    sellSweepBar = buySweepBar = -10**9
    eBT = eBB = np.nan; eBbar = -1; eBdone = False
    eST = eSB = np.nan; eSbar = -1; eSdone = False
    pos = None
    trades = []

    def close(p, px, res):
        d, entry, risk = p["dir"], p["entry"], p["risk"]
        rr = d * (px - entry) / risk
        total_r = p["banked"] + p["size"] * rr
        trades.append({"dir": d, "entry_time": p["entry_time"], "year": p["year"],
                       "type": p["type"], "risk": risk, "result": res,
                       "r": total_r, "points": total_r * risk})

    for i in range(n):
        # ---- manage open position ----
        if pos is not None:
            d, entry, risk = pos["dir"], pos["entry"], pos["risk"]
            exited = False
            # 1) stop, using stop in force at bar start
            if d == 1 and l[i] <= pos["sl"]:
                close(pos, pos["sl"], "SL"); pos = None; exited = True
            elif d == -1 and h[i] >= pos["sl"]:
                close(pos, pos["sl"], "SL"); pos = None; exited = True
            if not exited:
                # 2) arm / scale / trail at +1R (affects this bar's target & later stops)
                rExt = (h[i] - entry) / risk if d == 1 else (entry - l[i]) / risk
                if rExt >= 1.0 and not pos["armed"]:
                    if use_be:
                        pos["sl"] = entry; pos["armed"] = True
                    elif use_scale:
                        pos["banked"] += 0.5 * 1.0
                        pos["size"] = 0.5
                        pos["sl"] = entry
                        pos["armed"] = True
                    elif use_trail:
                        pos["armed"] = True
                if use_trail and pos["armed"]:
                    pos["sl"] = max(pos["sl"], h[i] - risk) if d == 1 \
                        else min(pos["sl"], l[i] + risk)
                # 3) fixed target
                if pos["tp"] is not None:
                    if d == 1 and h[i] >= pos["tp"]:
                        close(pos, pos["tp"], "TP"); pos = None; exited = True
                    elif d == -1 and l[i] <= pos["tp"]:
                        close(pos, pos["tp"], "TP"); pos = None; exited = True
            if not exited and pos is not None and mins[i] >= EOD_MIN:
                close(pos, c[i], "EOD"); pos = None

        # ---- 1M structure / sweeps ----
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
                if not (MAX_RISK > 0 and risk > MAX_RISK) and flat:
                    tp = None if use_trail else entry + tgt * risk
                    rec = {"dir": 1, "entry": entry, "sl": sl, "tp": tp, "risk": risk,
                           "size": 1.0, "banked": 0.0, "armed": False,
                           "entry_time": P["df"].index[i], "year": year[i],
                           "type": "continuation" if rb == 1 else "neutral"}
                    if l[i] <= sl:
                        close(rec, sl, "SL")
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
                if not (MAX_RISK > 0 and risk > MAX_RISK) and flat:
                    tp = None if use_trail else entry - tgt * risk
                    rec = {"dir": -1, "entry": entry, "sl": sl, "tp": tp, "risk": risk,
                           "size": 1.0, "banked": 0.0, "armed": False,
                           "entry_time": P["df"].index[i], "year": year[i],
                           "type": "continuation" if rb == -1 else "neutral"}
                    if h[i] >= sl:
                        close(rec, sl, "SL")
                    else:
                        pos = rec

    if pos is not None:
        close(pos, c[-1], "EOD")
    return pd.DataFrame(trades)


def stats(tr):
    r = tr.r.to_numpy()
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    pf = pos / neg if neg else float("inf")
    return dict(n=len(tr), win=(r > 0).mean() * 100, totR=r.sum(), avgR=r.mean(),
                pf=pf, dd=dd, rdd=r.sum() / abs(dd) if dd else float("inf"))


def main():
    print("Precomputing HTF engines (once)...")
    P = precompute()
    modes = ["baseline", "be_1r", "scale_1r", "trail_1r", "target_3r", "be_1r_3r"]
    desc = {"baseline": "fixed 2R / -1R (shipped)",
            "be_1r": "2R target, breakeven stop at +1R",
            "scale_1r": "half off at +1R -> BE, rest to 2R",
            "trail_1r": "trail 1R behind extreme after +1R",
            "target_3r": "fixed 3R / -1R (let it run)",
            "be_1r_3r": "3R target, breakeven stop at +1R"}
    print(f"\nSession 09:30-13:00 ET | levers mit=5 frac=1.0 | NQ 2022-2025\n")
    print(f"{'mode':<12}{'':<34}{'n':>4} {'win%':>6} {'totR':>8} {'avgR':>7} {'PF':>6} {'maxDD':>7} {'R/DD':>6}")
    print("-" * 92)
    results = {}
    for m in modes:
        tr = simulate_exits(P, m)
        results[m] = tr
        s = stats(tr)
        print(f"{m:<12}{desc[m]:<34}{s['n']:>4} {s['win']:>6.1f} {s['totR']:>+8.1f} "
              f"{s['avgR']:>+7.3f} {s['pf']:>6.2f} {s['dd']:>+7.1f} {s['rdd']:>6.1f}")
    print("-" * 92)
    # per-year for the two best-by-PF and best-by-totR, plus exit mix
    base_tot = stats(results["baseline"])["totR"]
    print("\nPer-year (totR) vs baseline, and exit mix:")
    for m in modes:
        tr = results[m]
        yr = {int(y): round(float(tr[tr.year == y].r.sum()), 1)
              for y in sorted(tr.year.unique())}
        mix = tr.result.value_counts().to_dict()
        d = stats(tr)["totR"] - base_tot
        print(f"  {m:<11} {yr}  d_vs_base={d:+6.1f}R  exits={mix}")


if __name__ == "__main__":
    main()
