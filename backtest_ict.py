#!/usr/bin/env python3
"""
Backtest of the ICT NAS100 multi-timeframe strategy on NQ 1-minute data.

Mirrors the logic of ICT_NAS100_Indicator.pine:
  4H  : FVG mitigation -> rejection -> MSS  => bias (bull/bear/neutral)
  15M : pivot-MSS order flow + FVG-of-direction mitigation (with freshness age)
  1M  : MSS (mandatory) + 1M FVG formed while aligned = entry trigger
  Neutral 4H bias: requires 4H FVG boundary touch + liquidity sweep (reversal)
  Kill zone: entries only 09:15-12:00 ET

Trade rules (Part 5 of the strategy doc):
  Long : entry at top of 1M bull FVG, SL below FVG bottom - buffer, TP = 2.5R
  Short: entry at bottom of 1M bear FVG, SL above FVG top + buffer, TP = 2.5R
  Max 2 trades per day. Open trades force-closed at 15:59 ET.
  Conservative fills: if SL and TP are both touched in one bar, SL wins;
  if the entry bar itself trades through SL, the trade is an immediate loss.

HTF values use only *completed* 4H/15M bars (no lookahead, no repaint).
"""
import sys
import numpy as np
import pandas as pd

CSV = "/Users/manjunathb/Downloads/Dataset_NQ_1min_2022_2025.csv"

# Parameters (match the Pine indicator defaults)
PIV_BIAS = 3
PIV_FLOW = 3
PIV_CHART = 3
MSS_MAX_AGE = 30      # 1m bars
MIT_MAX_AGE = 40      # 15m bars
SWEEP_LEN = 20        # 1m bars
REV_MAX_AGE = 60      # 1m bars
KZ_START = 9 * 60 + 15   # 09:15 ET
KZ_END = 12 * 60         # 12:00 ET
EOD_MIN = 15 * 60 + 59   # force close at 15:59 ET bar
SL_BUFFER = 2.0          # points beyond the FVG
RR = 2.5
MAX_TRADES_PER_DAY = 2


def load_data():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    df["ts"] = pd.to_datetime(df["timestamp ET"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    df = df[["open", "high", "low", "close"]].astype(float).dropna()
    return df


def resample(df, rule):
    r = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return r


def rolling_max(a, n):
    s = pd.Series(a).rolling(n, min_periods=1).max().to_numpy()
    return s


def rolling_min(a, n):
    s = pd.Series(a).rolling(n, min_periods=1).min().to_numpy()
    return s


def bias_engine(o, h, l, c, piv):
    """4H engine. Returns per-bar arrays: bias, bullTop, bullBot, bearTop, bearBot."""
    n = len(c)
    hh = rolling_max(h, piv * 2 + 1)
    ll = rolling_min(l, piv * 2 + 1)
    bias = np.zeros(n, dtype=int)
    bT = np.full(n, np.nan); bB = np.full(n, np.nan)
    sT = np.full(n, np.nan); sB = np.full(n, np.nan)

    bullTop = bullBot = bearTop = bearBot = np.nan
    bullMit = bullRej = bearMit = bearRej = False
    bullTrig = bearTrig = np.nan
    b = 0
    for i in range(n):
        if i >= 2:
            if l[i] > h[i - 2]:  # bullish FVG confirmed
                bullTop, bullBot = l[i], h[i - 2]
                bullMit = bullRej = False
                bullTrig = np.nan
            if h[i] < l[i - 2]:  # bearish FVG confirmed
                bearTop, bearBot = l[i - 2], h[i]
                bearMit = bearRej = False
                bearTrig = np.nan
        if not np.isnan(bullTop):
            if c[i] < bullBot:
                bullTop = bullBot = np.nan
                bullMit = bullRej = False
                if b == 1:
                    b = 0
            else:
                if not bullMit and l[i] <= bullTop:
                    bullMit = True
                    bullTrig = hh[i - 1] if i > 0 else h[i]
                if bullMit and not bullRej and c[i] > bullTop:
                    bullRej = True
                if bullMit and bullRej and not np.isnan(bullTrig) and c[i] > bullTrig:
                    b = 1
        if not np.isnan(bearTop):
            if c[i] > bearTop:
                bearTop = bearBot = np.nan
                bearMit = bearRej = False
                if b == -1:
                    b = 0
            else:
                if not bearMit and h[i] >= bearBot:
                    bearMit = True
                    bearTrig = ll[i - 1] if i > 0 else l[i]
                if bearMit and not bearRej and c[i] < bearBot:
                    bearRej = True
                if bearMit and bearRej and not np.isnan(bearTrig) and c[i] < bearTrig:
                    b = -1
        bias[i] = b
        bT[i], bB[i], sT[i], sB[i] = bullTop, bullBot, bearTop, bearBot
    return bias, bT, bB, sT, sB


def flow_engine(o, h, l, c, piv):
    """15M engine. Returns per-bar arrays: flow, bullMitAge, bearMitAge."""
    n = len(c)
    flow = np.zeros(n, dtype=int)
    bAge = np.full(n, -1, dtype=int)
    sAge = np.full(n, -1, dtype=int)

    lastPh = lastPl = np.nan
    f = 0
    bullTop = bullBot = bearTop = bearBot = np.nan
    bullMitAge = bearMitAge = -1
    for i in range(n):
        # pivot high/low confirmed at i for the bar at i-piv
        if i >= 2 * piv:
            p = i - piv
            seg_h = h[i - 2 * piv:i + 1]
            if h[p] == seg_h.max() and (seg_h == h[p]).sum() == 1:
                lastPh = h[p]
            seg_l = l[i - 2 * piv:i + 1]
            if l[p] == seg_l.min() and (seg_l == l[p]).sum() == 1:
                lastPl = l[p]
        if not np.isnan(lastPh) and c[i] > lastPh:
            f = 1
            lastPh = np.nan
        if not np.isnan(lastPl) and c[i] < lastPl:
            f = -1
            lastPl = np.nan

        if i >= 2:
            if l[i] > h[i - 2]:
                bullTop, bullBot = l[i], h[i - 2]
                bullMitAge = -1
            if h[i] < l[i - 2]:
                bearTop, bearBot = l[i - 2], h[i]
                bearMitAge = -1
        if not np.isnan(bullTop):
            if c[i] < bullBot:
                bullTop = bullBot = np.nan
                bullMitAge = -1
            elif bullMitAge < 0:
                if l[i] <= bullTop:
                    bullMitAge = 0
            else:
                bullMitAge += 1
        if not np.isnan(bearTop):
            if c[i] > bearTop:
                bearTop = bearBot = np.nan
                bearMitAge = -1
            elif bearMitAge < 0:
                if h[i] >= bearBot:
                    bearMitAge = 0
            else:
                bearMitAge += 1
        flow[i] = f
        bAge[i] = bullMitAge
        sAge[i] = bearMitAge
    return flow, bAge, sAge


def main():
    df = load_data()
    print(f"Loaded {len(df):,} 1m bars: {df.index[0]} -> {df.index[-1]}")

    df4 = resample(df, "4h")
    df15 = resample(df, "15min")
    print(f"4H bars: {len(df4):,}   15M bars: {len(df15):,}")

    h4bias, h4bT, h4bB, h4sT, h4sB = bias_engine(
        df4["open"].to_numpy(), df4["high"].to_numpy(),
        df4["low"].to_numpy(), df4["close"].to_numpy(), PIV_BIAS)
    m15flow, m15bAge, m15sAge = flow_engine(
        df15["open"].to_numpy(), df15["high"].to_numpy(),
        df15["low"].to_numpy(), df15["close"].to_numpy(), PIV_FLOW)

    # map each 1m bar -> last COMPLETED HTF bar (close time <= 1m bar open)
    t1 = df.index.asi8
    close4 = (df4.index + pd.Timedelta(hours=4)).asi8
    close15 = (df15.index + pd.Timedelta(minutes=15)).asi8
    idx4 = np.searchsorted(close4, t1, side="right") - 1
    idx15 = np.searchsorted(close15, t1, side="right") - 1

    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    n = len(df)
    mins = df.index.hour * 60 + df.index.minute
    mins = np.asarray(mins)
    dates = df.index.normalize().asi8

    prevLo = pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy()
    prevHi = pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy()

    # 1m state
    lastPh = lastPl = np.nan
    mssDir = 0; mssBar = -10**9
    sellSweepBar = buySweepBar = -10**9
    h4BullTouchBar = h4BearTouchBar = -10**9
    eBullTop = eBullBot = np.nan; eBullBar = -1; eBullDone = False
    eBearTop = eBearBot = np.nan; eBearBar = -1; eBearDone = False

    pos = None  # dict when a trade is open
    trades = []
    day_key = None; day_count = 0

    for i in range(n):
        if dates[i] != day_key:
            day_key = dates[i]; day_count = 0

        # ---- manage open position first (bar-by-bar, SL priority) ----
        if pos is not None:
            exit_price = None; result = None
            if pos["dir"] == 1:
                if l[i] <= pos["sl"]:
                    exit_price, result = pos["sl"], "SL"
                elif h[i] >= pos["tp"]:
                    exit_price, result = pos["tp"], "TP"
            else:
                if h[i] >= pos["sl"]:
                    exit_price, result = pos["sl"], "SL"
                elif l[i] <= pos["tp"]:
                    exit_price, result = pos["tp"], "TP"
            if exit_price is None and mins[i] >= EOD_MIN:
                exit_price, result = c[i], "EOD"
            if exit_price is not None:
                pts = (exit_price - pos["entry"]) * pos["dir"]
                trades.append({**pos, "exit_time": df.index[i], "exit": exit_price,
                               "result": result, "points": pts, "r": pts / pos["risk"]})
                pos = None

        # ---- 1m pivots -> MSS ----
        if i >= 2 * PIV_CHART:
            p = i - PIV_CHART
            seg = h[i - 2 * PIV_CHART:i + 1]
            if h[p] == seg.max() and (seg == h[p]).sum() == 1:
                lastPh = h[p]
            seg = l[i - 2 * PIV_CHART:i + 1]
            if l[p] == seg.min() and (seg == l[p]).sum() == 1:
                lastPl = l[p]
        if not np.isnan(lastPh) and c[i] > lastPh:
            mssDir = 1; mssBar = i; lastPh = np.nan
        if not np.isnan(lastPl) and c[i] < lastPl:
            mssDir = -1; mssBar = i; lastPl = np.nan
        mssRecent = (i - mssBar) <= MSS_MAX_AGE

        # ---- sweeps ----
        if not np.isnan(prevLo[i]):
            if l[i] < prevLo[i] and c[i] > prevLo[i]:
                sellSweepBar = i
            if h[i] > prevHi[i] and c[i] < prevHi[i]:
                buySweepBar = i

        # ---- HTF state (completed bars only) ----
        j4 = idx4[i]; j15 = idx15[i]
        if j4 < 0 or j15 < 0:
            continue
        bias = h4bias[j4]
        bullT4, bearB4 = h4bT[j4], h4sB[j4]
        flow = m15flow[j15]
        mitL = 0 <= m15bAge[j15] <= MIT_MAX_AGE
        mitS = 0 <= m15sAge[j15] <= MIT_MAX_AGE

        if not np.isnan(bullT4) and l[i] <= bullT4:
            h4BullTouchBar = i
        if not np.isnan(bearB4) and h[i] >= bearB4:
            h4BearTouchBar = i

        neutralLong = bias == 0 and (i - h4BullTouchBar) <= REV_MAX_AGE and (i - sellSweepBar) <= REV_MAX_AGE
        neutralShort = bias == 0 and (i - h4BearTouchBar) <= REV_MAX_AGE and (i - buySweepBar) <= REV_MAX_AGE
        setupLong = (bias == 1 or neutralLong) and flow == 1 and mitL and mssDir == 1 and mssRecent
        setupShort = (bias == -1 or neutralShort) and flow == -1 and mitS and mssDir == -1 and mssRecent

        # ---- entry FVG registration ----
        if i >= 2:
            if l[i] > h[i - 2] and setupLong:
                eBullTop, eBullBot, eBullBar, eBullDone = l[i], h[i - 2], i, False
            if h[i] < l[i - 2] and setupShort:
                eBearTop, eBearBot, eBearBar, eBearDone = l[i - 2], h[i], i, False
        if not np.isnan(eBullTop) and c[i] < eBullBot:
            eBullTop = eBullBot = np.nan
        if not np.isnan(eBearTop) and c[i] > eBearTop:
            eBearTop = eBearBot = np.nan

        # ---- triggers ----
        inKZ = KZ_START <= mins[i] < KZ_END
        can_enter = pos is None and inKZ and day_count < MAX_TRADES_PER_DAY

        if can_enter and setupLong and not np.isnan(eBullTop) and not eBullDone \
                and i > eBullBar and l[i] <= eBullTop:
            entry = eBullTop
            sl = eBullBot - SL_BUFFER
            risk = entry - sl
            tp = entry + RR * risk
            eBullDone = True
            day_count += 1
            ttype = "continuation" if bias == 1 else "reversal"
            if l[i] <= sl:  # entry bar traded through the stop
                trades.append({"dir": 1, "entry_time": df.index[i], "entry": entry,
                               "sl": sl, "tp": tp, "risk": risk, "type": ttype,
                               "exit_time": df.index[i], "exit": sl, "result": "SL",
                               "points": sl - entry, "r": -1.0})
            else:
                pos = {"dir": 1, "entry_time": df.index[i], "entry": entry,
                       "sl": sl, "tp": tp, "risk": risk, "type": ttype}

        elif can_enter and setupShort and not np.isnan(eBearTop) and not eBearDone \
                and i > eBearBar and h[i] >= eBearBot:
            entry = eBearBot
            sl = eBearTop + SL_BUFFER
            risk = sl - entry
            tp = entry - RR * risk
            eBearDone = True
            day_count += 1
            ttype = "continuation" if bias == -1 else "reversal"
            if h[i] >= sl:
                trades.append({"dir": -1, "entry_time": df.index[i], "entry": entry,
                               "sl": sl, "tp": tp, "risk": risk, "type": ttype,
                               "exit_time": df.index[i], "exit": sl, "result": "SL",
                               "points": entry - sl, "r": -1.0})
            else:
                pos = {"dir": -1, "entry_time": df.index[i], "entry": entry,
                       "sl": sl, "tp": tp, "risk": risk, "type": ttype}

    # close any dangling position at the last bar
    if pos is not None:
        pts = (c[-1] - pos["entry"]) * pos["dir"]
        trades.append({**pos, "exit_time": df.index[-1], "exit": c[-1],
                       "result": "EOD", "points": pts, "r": pts / pos["risk"]})

    tr = pd.DataFrame(trades)
    if tr.empty:
        print("No trades generated.")
        return
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    tr["year"] = tr["entry_time"].dt.year
    tr["cum_r"] = tr["r"].cumsum()
    out = "/Users/manjunathb/Documents/FAB/ICT Indicator/backtest_trades.csv"
    tr.to_csv(out, index=False)

    # ---- stats ----
    def block(t, label):
        nT = len(t)
        wins = (t["r"] > 0).sum()
        wr = wins / nT * 100 if nT else 0
        tot = t["r"].sum()
        avg = t["r"].mean()
        pos_r = t.loc[t["r"] > 0, "r"].sum()
        neg_r = -t.loc[t["r"] < 0, "r"].sum()
        pf = pos_r / neg_r if neg_r > 0 else float("inf")
        eq = t["r"].cumsum()
        dd = (eq - eq.cummax()).min()
        print(f"{label:<22} trades={nT:<5} win%={wr:5.1f}  totR={tot:+8.1f}  "
              f"avgR={avg:+6.3f}  PF={pf:4.2f}  maxDD_R={dd:6.1f}  pts={t['points'].sum():+10.1f}")

    print()
    print("=" * 100)
    block(tr, "ALL")
    for y in sorted(tr["year"].unique()):
        block(tr[tr["year"] == y], f"  {y}")
    for d, lab in [(1, "  Longs"), (-1, "  Shorts")]:
        block(tr[tr["dir"] == d], lab)
    for ty in ["continuation", "reversal"]:
        sub = tr[tr["type"] == ty]
        if len(sub):
            block(sub, f"  {ty.capitalize()}")
    print("=" * 100)
    print("\nExit breakdown:", tr["result"].value_counts().to_dict())
    print(f"Avg risk per trade: {tr['risk'].mean():.1f} pts | "
          f"median: {tr['risk'].median():.1f} pts")
    print(f"Trades CSV: {out}")


if __name__ == "__main__":
    main()
