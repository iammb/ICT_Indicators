#!/usr/bin/env python3
"""
Parameter sweep for the ICT NQ backtest — hunting for >50% win rate
without giving up positive expectancy.

Levers tested (vs backtest_ict.py baseline):
  - continuation-only (drop the money-losing neutral-bias reversals)
  - RR target: 1.0 / 1.25 / 1.5 / 2.0 / 2.5
  - stop placement: FVG-based vs swing-based (doc: beyond the MSS swing)
  - breakeven stop move once +1R is reached
  - minimum entry-FVG height (quality filter)
  - longs-only
"""
import numpy as np
import pandas as pd
from backtest_ict import (load_data, resample, bias_engine, flow_engine,
                          PIV_BIAS, PIV_FLOW, PIV_CHART, MSS_MAX_AGE,
                          MIT_MAX_AGE, SWEEP_LEN, REV_MAX_AGE,
                          KZ_START, KZ_END, EOD_MIN, SL_BUFFER,
                          MAX_TRADES_PER_DAY)


def run(df, htf, cfg, h4flow=None):
    (h4bias, h4bT, h4sB, m15flow, m15bAge, m15sAge, idx4, idx15) = htf
    neutral_flow = cfg.get("neutral_flow", False)   # neutral 4H: trade with 15M flow
    h4_fallback = cfg.get("h4_fallback", False)     # neutral 4H: use 4H structure dir
    max_risk = cfg.get("max_risk", 0.0)             # cap on stop distance in points (0 = off)
    fvg_fallback = cfg.get("fvg_fallback", False)   # too-wide swing stop -> try FVG stop
    mit_max = cfg.get("mit_age", MIT_MAX_AGE)       # 15M FVG mitigation freshness
    mss_max = cfg.get("mss_age", MSS_MAX_AGE)       # 1M MSS freshness
    max_trades = cfg.get("max_trades", MAX_TRADES_PER_DAY)
    disp_sweep = cfg.get("disp_sweep", 0.0)         # displacement body >= x*ATR counts as sweep
    entry_close = cfg.get("entry_close", False)     # enter at close of FVG bar (no retrace wait)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    n = len(df)
    mins = np.asarray(df.index.hour * 60 + df.index.minute)
    dates = df.index.normalize().asi8
    prevLo = pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy()
    prevHi = pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy()

    rr = cfg["rr"]
    cont_only = cfg.get("cont_only", True)
    sl_swing = cfg.get("sl_swing", False)
    be_1r = cfg.get("be_1r", False)
    min_fvg = cfg.get("min_fvg", 0.0)
    longs_only = cfg.get("longs_only", False)
    disp_mult = cfg.get("disp_mult", 0.0)   # displacement: middle-candle body >= mult*ATR14
    pd_len = cfg.get("pd_len", 0)           # premium/discount dealing range length (0 = off)
    sweep_all = cfg.get("sweep_all", False) # require liquidity sweep before ALL entries
    kz_start = cfg.get("kz_start", KZ_START)
    kz_end = cfg.get("kz_end", KZ_END)

    tr_ = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr_[0] = h[0] - l[0]
    atr = pd.Series(tr_).rolling(14).mean().to_numpy()
    body = np.abs(c - o)
    if pd_len > 0:
        dealHi = pd.Series(h).rolling(pd_len).max().shift(1).to_numpy()
        dealLo = pd.Series(l).rolling(pd_len).min().shift(1).to_numpy()
        eqArr = (dealHi + dealLo) / 2.0
    else:
        eqArr = None

    lastPh = lastPl = np.nan
    lastSwingHi = lastSwingLo = np.nan   # most recent confirmed pivots (persistent)
    mssDir = 0; mssBar = -10**9
    sellSweepBar = buySweepBar = -10**9
    h4BullTouchBar = h4BearTouchBar = -10**9
    eBullTop = eBullBot = np.nan; eBullBar = -1; eBullDone = False
    eBearTop = eBearBot = np.nan; eBearBar = -1; eBearDone = False
    pos = None
    rs = []; results = []; years = []
    day_key = None; day_count = 0

    for i in range(n):
        if dates[i] != day_key:
            day_key = dates[i]; day_count = 0

        if pos is not None:
            exit_price = None; res = None
            if pos["dir"] == 1:
                if l[i] <= pos["sl"]:
                    exit_price, res = pos["sl"], "SL"
                elif h[i] >= pos["tp"]:
                    exit_price, res = pos["tp"], "TP"
                elif be_1r and not pos["be"] and h[i] >= pos["entry"] + pos["risk"]:
                    pos["sl"] = pos["entry"]; pos["be"] = True
            else:
                if h[i] >= pos["sl"]:
                    exit_price, res = pos["sl"], "SL"
                elif l[i] <= pos["tp"]:
                    exit_price, res = pos["tp"], "TP"
                elif be_1r and not pos["be"] and l[i] <= pos["entry"] - pos["risk"]:
                    pos["sl"] = pos["entry"]; pos["be"] = True
            if exit_price is None and mins[i] >= EOD_MIN:
                exit_price, res = c[i], "EOD"
            if exit_price is not None:
                pts = (exit_price - pos["entry"]) * pos["dir"]
                rs.append(pts / pos["risk"]); results.append(res)
                years.append(df.index[i].year)
                pos = None

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
        mssRecent = (i - mssBar) <= mss_max

        if not np.isnan(prevLo[i]):
            if l[i] < prevLo[i] and c[i] > prevLo[i]:
                sellSweepBar = i
            if h[i] > prevHi[i] and c[i] < prevHi[i]:
                buySweepBar = i
        if disp_sweep > 0 and not np.isnan(atr[i]) and body[i] >= disp_sweep * atr[i]:
            # strong displacement = institutional intent; grants the same
            # permission as a sweep in its direction
            if c[i] > o[i]:
                sellSweepBar = i
            else:
                buySweepBar = i

        j4 = idx4[i]; j15 = idx15[i]
        if j4 < 0 or j15 < 0:
            continue
        bias = h4bias[j4]
        if bias == 0 and h4_fallback and h4flow is not None:
            bias = h4flow[j4]
        bullT4, bearB4 = h4bT[j4], h4sB[j4]
        flow = m15flow[j15]
        mitL = 0 <= m15bAge[j15] <= mit_max
        mitS = 0 <= m15sAge[j15] <= mit_max

        if not np.isnan(bullT4) and l[i] <= bullT4:
            h4BullTouchBar = i
        if not np.isnan(bearB4) and h[i] >= bearB4:
            h4BearTouchBar = i

        neutralLong = (not cont_only) and bias == 0 and \
            (i - h4BullTouchBar) <= REV_MAX_AGE and (i - sellSweepBar) <= REV_MAX_AGE
        neutralShort = (not cont_only) and bias == 0 and \
            (i - h4BearTouchBar) <= REV_MAX_AGE and (i - buySweepBar) <= REV_MAX_AGE
        biasOkL = bias == 1 or neutralLong or (neutral_flow and bias == 0 and flow == 1)
        biasOkS = bias == -1 or neutralShort or (neutral_flow and bias == 0 and flow == -1)
        setupLong = biasOkL and flow == 1 and mitL and mssDir == 1 and mssRecent
        setupShort = biasOkS and flow == -1 and mitS and mssDir == -1 and mssRecent
        if longs_only:
            setupShort = False
        if sweep_all:
            setupLong = setupLong and (i - sellSweepBar) <= REV_MAX_AGE
            setupShort = setupShort and (i - buySweepBar) <= REV_MAX_AGE
        if eqArr is not None and not np.isnan(eqArr[i]):
            setupLong = setupLong and c[i] <= eqArr[i]     # buy in discount only
            setupShort = setupShort and c[i] >= eqArr[i]   # sell in premium only

        if i >= 2:
            dispOK = disp_mult <= 0 or (not np.isnan(atr[i - 1]) and body[i - 1] >= disp_mult * atr[i - 1])
            if l[i] > h[i - 2] and setupLong and (l[i] - h[i - 2]) >= min_fvg and dispOK:
                eBullTop, eBullBot, eBullBar, eBullDone = l[i], h[i - 2], i, False
            if h[i] < l[i - 2] and setupShort and (l[i - 2] - h[i]) >= min_fvg and dispOK:
                eBearTop, eBearBot, eBearBar, eBearDone = l[i - 2], h[i], i, False
        if not np.isnan(eBullTop) and c[i] < eBullBot:
            eBullTop = eBullBot = np.nan
        if not np.isnan(eBearTop) and c[i] > eBearTop:
            eBearTop = eBearBot = np.nan

        inKZ = kz_start <= mins[i] < kz_end
        can_enter = pos is None and inKZ and day_count < max_trades

        if entry_close and can_enter and not np.isnan(eBullTop) and not eBullDone and i == eBullBar:
            entry = c[i]
            base = min(eBullBot, lastSwingLo) if (sl_swing and not np.isnan(lastSwingLo)) else eBullBot
            sl = base - SL_BUFFER
            risk = entry - sl
            eBullDone = True
            if risk > 0 and not (max_risk > 0 and risk > max_risk):
                day_count += 1
                pos = {"dir": 1, "entry": entry, "sl": sl, "tp": entry + rr * risk, "risk": risk, "be": False}
            continue
        if entry_close and can_enter and not np.isnan(eBearTop) and not eBearDone and i == eBearBar:
            entry = c[i]
            base = max(eBearTop, lastSwingHi) if (sl_swing and not np.isnan(lastSwingHi)) else eBearTop
            sl = base + SL_BUFFER
            risk = sl - entry
            eBearDone = True
            if risk > 0 and not (max_risk > 0 and risk > max_risk):
                day_count += 1
                pos = {"dir": -1, "entry": entry, "sl": sl, "tp": entry - rr * risk, "risk": risk, "be": False}
            continue

        if not entry_close and can_enter and setupLong and not np.isnan(eBullTop) and not eBullDone \
                and i > eBullBar and l[i] <= eBullTop:
            entry = eBullTop
            base = min(eBullBot, lastSwingLo) if (sl_swing and not np.isnan(lastSwingLo)) else eBullBot
            sl = base - SL_BUFFER
            risk = entry - sl
            if max_risk > 0 and risk > max_risk:
                if fvg_fallback and (entry - (eBullBot - SL_BUFFER)) <= max_risk:
                    sl = eBullBot - SL_BUFFER
                    risk = entry - sl
                else:
                    eBullDone = True
                    continue
            tp = entry + rr * risk
            eBullDone = True; day_count += 1
            if l[i] <= sl:
                rs.append(-1.0); results.append("SL"); years.append(df.index[i].year)
            else:
                pos = {"dir": 1, "entry": entry, "sl": sl, "tp": tp, "risk": risk, "be": False}
        elif not entry_close and can_enter and setupShort and not np.isnan(eBearTop) and not eBearDone \
                and i > eBearBar and h[i] >= eBearBot:
            entry = eBearBot
            base = max(eBearTop, lastSwingHi) if (sl_swing and not np.isnan(lastSwingHi)) else eBearTop
            sl = base + SL_BUFFER
            risk = sl - entry
            if max_risk > 0 and risk > max_risk:
                if fvg_fallback and ((eBearTop + SL_BUFFER) - entry) <= max_risk:
                    sl = eBearTop + SL_BUFFER
                    risk = sl - entry
                else:
                    eBearDone = True
                    continue
            tp = entry - rr * risk
            eBearDone = True; day_count += 1
            if h[i] >= sl:
                rs.append(-1.0); results.append("SL"); years.append(df.index[i].year)
            else:
                pos = {"dir": -1, "entry": entry, "sl": sl, "tp": tp, "risk": risk, "be": False}

    r = np.array(rs)
    if len(r) == 0:
        return None
    wins = (r > 0).sum(); be = (r == 0).sum()
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(eq)).min()
    pos_r = r[r > 0].sum(); neg_r = -r[r < 0].sum()
    yr = pd.Series(r, index=years).groupby(level=0).sum().round(1).to_dict()
    return dict(n=len(r), win=wins / len(r) * 100,
                win_ex_be=wins / max(len(r) - be, 1) * 100, be=be,
                totR=r.sum(), avgR=r.mean(),
                pf=pos_r / neg_r if neg_r else float("inf"), dd=dd, yearly=yr)


def main():
    df = load_data()
    df4 = resample(df, "4h"); df15 = resample(df, "15min")
    h4bias, h4bT, h4bB, h4sT, h4sB = bias_engine(
        df4["open"].to_numpy(), df4["high"].to_numpy(),
        df4["low"].to_numpy(), df4["close"].to_numpy(), PIV_BIAS)
    m15flow, m15bAge, m15sAge = flow_engine(
        df15["open"].to_numpy(), df15["high"].to_numpy(),
        df15["low"].to_numpy(), df15["close"].to_numpy(), PIV_FLOW)
    t1 = df.index.asi8
    idx4 = np.searchsorted((df4.index + pd.Timedelta(hours=4)).asi8, t1, "right") - 1
    idx15 = np.searchsorted((df15.index + pd.Timedelta(minutes=15)).asi8, t1, "right") - 1
    htf = (h4bias, h4bT, h4sB, m15flow, m15bAge, m15sAge, idx4, idx15)

    configs = [
        ("baseline all-setups RR2.5", dict(rr=2.5, cont_only=False)),
        ("cont-only RR2.5",           dict(rr=2.5)),
        ("cont-only RR2.0",           dict(rr=2.0)),
        ("cont-only RR1.5",           dict(rr=1.5)),
        ("cont-only RR1.25",          dict(rr=1.25)),
        ("cont-only RR1.0",           dict(rr=1.0)),
        ("cont RR1.5 swingSL",        dict(rr=1.5, sl_swing=True)),
        ("cont RR1.0 swingSL",        dict(rr=1.0, sl_swing=True)),
        ("cont RR2.5 BE@1R",          dict(rr=2.5, be_1r=True)),
        ("cont RR1.5 BE@1R",          dict(rr=1.5, be_1r=True)),
        ("cont RR1.5 minFVG2",        dict(rr=1.5, min_fvg=2.0)),
        ("cont RR1.0 minFVG2",        dict(rr=1.0, min_fvg=2.0)),
        ("cont RR1.5 swingSL minFVG2", dict(rr=1.5, sl_swing=True, min_fvg=2.0)),
        ("cont RR1.0 swingSL minFVG2", dict(rr=1.0, sl_swing=True, min_fvg=2.0)),
        ("cont RR1.5 longs-only",     dict(rr=1.5, longs_only=True)),
    ]
    print(f"{'config':<28} {'n':>4} {'win%':>6} {'winXbe%':>8} {'BE':>3} "
          f"{'totR':>8} {'avgR':>7} {'PF':>5} {'maxDD':>6}  yearly R")
    print("-" * 110)
    for name, cfg in configs:
        s = run(df, htf, cfg)
        if s is None:
            print(f"{name:<28} no trades")
            continue
        print(f"{name:<28} {s['n']:>4} {s['win']:>6.1f} {s['win_ex_be']:>8.1f} {s['be']:>3} "
              f"{s['totR']:>+8.1f} {s['avgR']:>+7.3f} {s['pf']:>5.2f} {s['dd']:>6.1f}  {s['yearly']}")


if __name__ == "__main__":
    main()
