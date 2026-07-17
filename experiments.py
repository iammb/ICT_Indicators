#!/usr/bin/env python3
"""
Single-variable experiments on top of the independent backtest (backtest_v2).
Each config changes ONE thing vs the baseline so we can see what actually
moves win rate, and whether it helps or hurts expectancy (totR / PF).

Shared, config-independent arrays (HTF engines, 1M pivots, sweeps, ATR) are
computed once; only the fast per-bar sim loop re-runs per config.
"""
import numpy as np
import pandas as pd
from backtest_v2 import (load, resample, bias_engine, flow_engine,
                         strict_pivots, confirmed_index, CSV,
                         PIV_BIAS, PIV_FLOW, PIV_CHART, SWEEP_LEN, REV_MAX_AGE,
                         SL_BUF)

EOD_MIN = 16 * 60 + 59


def precompute():
    df = load(CSV)
    d4 = resample(df, "4h", "start_day")
    d15 = resample(df, "15min", "start_day")
    b4, b4bT, b4bB, b4sT, b4sB = bias_engine(d4.h.values, d4.l.values, d4.c.values, PIV_BIAS)
    f4, _, _ = flow_engine(d4.h.values, d4.l.values, d4.c.values, PIV_BIAS)
    f15, a15b, a15s = flow_engine(d15.h.values, d15.l.values, d15.c.values, PIV_FLOW)
    t = df.index.asi8
    i4 = confirmed_index(t, d4.index, pd.Timedelta(hours=4))
    i15 = confirmed_index(t, d15.index, pd.Timedelta(minutes=15))
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    mins = (df.index.hour * 60 + df.index.minute).to_numpy()
    phC, plC = strict_pivots(h, l, PIV_CHART)
    prevLo = pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy()
    prevHi = pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy()
    # ATR14 (Wilder) on 1M for the displacement filter
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                      np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().to_numpy()
    body = np.abs(c - o)
    return dict(df=df, o=o, h=h, l=l, c=c, mins=mins, phC=phC, plC=plC,
                prevLo=prevLo, prevHi=prevHi, atr=atr, body=body,
                b4=b4, b4bT=b4bT, b4sB=b4sB, f4=f4, f15=f15,
                a15b=a15b, a15s=a15s, i4=i4, i15=i15,
                nd4=len(d4), nd15=len(d15))


def sim(P, cfg):
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]
    mins, phC, plC = P["mins"], P["phC"], P["plC"]
    prevLo, prevHi, atr, body = P["prevLo"], P["prevHi"], P["atr"], P["body"]
    b4, b4bT, b4sB, f4 = P["b4"], P["b4bT"], P["b4sB"], P["f4"]
    f15, a15b, a15s, i4, i15 = P["f15"], P["a15b"], P["a15s"], P["i4"], P["i15"]
    df = P["df"]
    n = len(c)

    rr = cfg["rr"]; kz0, kz1 = cfg["kz"]; disp = cfg["disp"]
    cont_only = cfg["cont_only"]; be = cfg["be"]
    min_risk, max_risk = cfg["min_risk"], cfg["max_risk"]
    mit_age, mss_age = cfg["mit_age"], cfg["mss_age"]

    lastPh = lastPl = swingHi = swingLo = np.nan
    mssDir, mssBar = 0, -10**9
    sellB = buyB = -10**9
    eBT = eBB = np.nan; eBbar = -1; eBdone = False
    eST = eSB = np.nan; eSbar = -1; eSdone = False
    pos = None
    trades = []

    for i in range(n):
        if pos is not None:
            px = res = None
            if pos["dir"] == 1:
                if l[i] <= pos["sl"]: px, res = pos["sl"], ("BE" if pos.get("moved") and pos["sl"] >= pos["entry"] else "SL")
                elif h[i] >= pos["tp"]: px, res = pos["tp"], "TP"
            else:
                if h[i] >= pos["sl"]: px, res = pos["sl"], ("BE" if pos.get("moved") and pos["sl"] <= pos["entry"] else "SL")
                elif l[i] <= pos["tp"]: px, res = pos["tp"], "TP"
            if px is None and mins[i] >= EOD_MIN:
                px, res = c[i], "EOD"
            if px is not None:
                pts = (px - pos["entry"]) * pos["dir"]
                trades.append({**pos, "exit": px, "result": res,
                               "points": pts, "r": pts / pos["risk"]})
                pos = None
            elif be and not pos.get("moved"):
                one_r = pos["risk"]
                if pos["dir"] == 1 and h[i] >= pos["entry"] + one_r:
                    pos["sl"] = pos["entry"]; pos["moved"] = True
                elif pos["dir"] == -1 and l[i] <= pos["entry"] - one_r:
                    pos["sl"] = pos["entry"]; pos["moved"] = True

        if not np.isnan(phC[i]): lastPh = swingHi = phC[i]
        if not np.isnan(plC[i]): lastPl = swingLo = plC[i]
        if not np.isnan(lastPh) and c[i] > lastPh: mssDir, mssBar, lastPh = 1, i, np.nan
        if not np.isnan(lastPl) and c[i] < lastPl: mssDir, mssBar, lastPl = -1, i, np.nan
        mssRecent = (i - mssBar) <= mss_age

        if not np.isnan(prevLo[i]):
            if l[i] < prevLo[i] and c[i] > prevLo[i]: sellB = i
            if h[i] > prevHi[i] and c[i] < prevHi[i]: buyB = i

        j4, j15 = i4[i], i15[i]
        if j4 < 0 or j15 < 0:
            continue
        raw = b4[j4]
        bias = raw if (cont_only or raw != 0) else f4[j4]
        flow = f15[j15]
        mitL = 0 <= a15b[j15] <= mit_age
        mitS = 0 <= a15s[j15] <= mit_age

        setupLong = (bias == 1 and flow == 1 and mitL and mssDir == 1 and mssRecent
                     and (i - sellB) <= REV_MAX_AGE)
        setupShort = (bias == -1 and flow == -1 and mitS and mssDir == -1 and mssRecent
                      and (i - buyB) <= REV_MAX_AGE)

        dispOK = disp <= 0 or (i >= 1 and body[i - 1] >= disp * atr[i - 1])
        if i >= 2:
            if l[i] > h[i - 2] and setupLong and dispOK:
                eBT, eBB, eBbar, eBdone = l[i], h[i - 2], i, False
            if h[i] < l[i - 2] and setupShort and dispOK:
                eST, eSB, eSbar, eSdone = l[i - 2], h[i], i, False
        if not np.isnan(eBT) and c[i] < eBB: eBT = eBB = np.nan
        if not np.isnan(eST) and c[i] > eST: eST = eSB = np.nan

        inKZ = kz0 <= mins[i] < kz1
        flat = pos is None

        if setupLong and not np.isnan(eBT) and not eBdone and inKZ and i > eBbar and l[i] <= eBT:
            eBdone = True
            base = min(eBB, swingLo) if not np.isnan(swingLo) else eBB
            sl = base - SL_BUF; entry = eBT
            if min_risk > 0: sl = min(sl, entry - min_risk)
            risk = entry - sl
            if not (max_risk > 0 and risk > max_risk) and flat:
                rec = {"dir": 1, "entry_time": df.index[i], "entry": entry, "sl": sl,
                       "tp": entry + rr * risk, "risk": risk, "moved": False,
                       "type": "cont" if raw == 1 else "neutral"}
                if l[i] <= sl:
                    trades.append({**rec, "exit": sl, "result": "SL",
                                   "points": sl - entry, "r": -1.0})
                else:
                    pos = rec
        elif setupShort and not np.isnan(eST) and not eSdone and inKZ and i > eSbar and h[i] >= eSB:
            eSdone = True
            base = max(eST, swingHi) if not np.isnan(swingHi) else eST
            sl = base + SL_BUF; entry = eSB
            if min_risk > 0: sl = max(sl, entry + min_risk)
            risk = sl - entry
            if not (max_risk > 0 and risk > max_risk) and flat:
                rec = {"dir": -1, "entry_time": df.index[i], "entry": entry, "sl": sl,
                       "tp": entry - rr * risk, "risk": risk, "moved": False,
                       "type": "cont" if raw == -1 else "neutral"}
                if h[i] >= sl:
                    trades.append({**rec, "exit": sl, "result": "SL",
                                   "points": entry - sl, "r": -1.0})
                else:
                    pos = rec

    return pd.DataFrame(trades)


def summarize(tr):
    n = len(tr)
    if n == 0:
        return dict(n=0)
    wins = (tr.r > 0).sum()
    losses = (tr.r < 0).sum()
    pos_r = tr.loc[tr.r > 0, "r"].sum()
    neg_r = -tr.loc[tr.r < 0, "r"].sum()
    eq = tr.r.cumsum()
    return dict(n=n, win=wins / n * 100, loss=losses / n * 100,
                totR=tr.r.sum(), avgR=tr.r.mean(),
                pf=pos_r / neg_r if neg_r else float("inf"),
                dd=(eq - eq.cummax()).min(), pts=tr.points.sum())


BASE = dict(rr=2.0, kz=(555, 720), disp=0.0, cont_only=False, be=False,
            min_risk=20.0, max_risk=55.0, mit_age=40, mss_age=30)


def cfg(**kw):
    d = dict(BASE); d.update(kw); return d


EXPERIMENTS = [
    ("BASELINE (rr2, all filters default)",         cfg()),
    ("TP target 1.5R",                              cfg(rr=1.5)),
    ("TP target 1.0R",                              cfg(rr=1.0)),
    ("TP target 3.0R",                              cfg(rr=3.0)),
    ("Breakeven stop after +1R",                    cfg(be=True)),
    ("Continuation-only (drop neutral days)",       cfg(cont_only=True)),
    ("Displacement filter body>=1.0xATR",           cfg(disp=1.0)),
    ("Displacement filter body>=1.5xATR",           cfg(disp=1.5)),
    ("Kill zone 09:30-11:00 only",                  cfg(kz=(570, 660))),
    ("Fresher 15M mitigation (age<=15)",            cfg(mit_age=15)),
    ("Fresher 1M MSS (age<=12)",                    cfg(mss_age=12)),
    ("Tighter max stop 40pt",                       cfg(max_risk=40.0)),
    ("COMBO: cont-only + disp1.0 + KZ0930-1100",    cfg(cont_only=True, disp=1.0, kz=(570, 660))),
    ("COMBO + TP 1.5R",                             cfg(cont_only=True, disp=1.0, kz=(570, 660), rr=1.5)),
]


def main():
    P = precompute()
    print(f"{'config':<44}{'n':>5}{'win%':>7}{'totR':>8}{'avgR':>8}{'PF':>6}{'maxDD':>8}")
    print("-" * 86)
    for name, c in EXPERIMENTS:
        s = summarize(sim(P, c))
        if s["n"] == 0:
            print(f"{name:<44}   no trades"); continue
        print(f"{name:<44}{s['n']:>5}{s['win']:>7.1f}{s['totR']:>+8.1f}"
              f"{s['avgR']:>+8.3f}{s['pf']:>6.2f}{s['dd']:>8.1f}")


if __name__ == "__main__":
    main()
