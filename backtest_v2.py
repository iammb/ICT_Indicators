#!/usr/bin/env python3
"""
Independent backtest of ICT_NAS100_Indicator.pine on NQ 1-minute data.

Written from scratch off the Pine source (not from backtest_ict.py). Purpose:
a second, independently-derived implementation of the SAME indicator so its
result can be trusted (or a discrepancy caught) rather than taken on faith.

Indicator logic reproduced (all Pine defaults):
  4H  biasEngine  : FVG mitigation -> rejection -> MSS  => bias {+1,-1,0}
  15M flowEngine  : pivot-MSS order flow + directional FVG mitigation age
  4H  flowEngine  : plain 4H market structure (neutral-bias fallback)
  1M  : MSS (mandatory) + 1M FVG that forms while aligned = armed entry zone
  Filters: NY morning session 09:30-13:00 ET, liquidity sweep required before every
           entry, 4H-structure fallback on neutral 4H bias (longs only - see
           NEUTRAL_LONG_ONLY: neutral-bias shorts backtested as a net loser).
  Plan   : entry = full fill through the 1M FVG (ENTRY_FRAC=1.0), stop beyond
           min(FVG, 1M swing) +/- 2pt buffer, min stop 20pt / max stop 55pt (these
           scale with price when USE_VOLSTOP is on - see below), take-profit at 2R.
           15M mitigation must be fresh (MIT_MAX_AGE=5 bars).
  HTF values are read from the last CONFIRMED 4H / 15M bar (htfConfirmed=on).

Trade-management layer (NOT in the indicator - my own, deliberately simple and
stated up front so the numbers are reproducible):
  * one position at a time; every signal that fires while flat is taken
    (no arbitrary trades-per-day cap)
  * fills are conservative: entry is a limit at the FVG edge; if the trigger
    bar's range also spans the stop, it's booked as an immediate loss; on any
    later bar that spans both stop and target, the STOP is assumed hit first
  * any position still open at 16:59 ET is force-closed at that bar's close
Independence note: this differs on purpose from backtest_ict.py, which capped
at 2 trades/day and closed at 15:59. Convergence of the headline edge across
the two designs is the actual cross-check.
"""
import numpy as np
import pandas as pd

CSV = "/Users/manjunathb/Downloads/Dataset_NQ_1min_2022_2025.csv"

# ---- indicator defaults (ICT_NAS100_Indicator.pine) ----
PIV_BIAS = PIV_FLOW = PIV_CHART = 3
MSS_MAX_AGE = 30       # 1M bars
MIT_MAX_AGE = 5        # 15M bars - fresh mitigation only (was 40; backtest: 46.9% win/PF 1.73
                       # vs 43.1%/1.47 at 40 - acting on the reaction NOW beats a stale tap)
SWEEP_LEN = 20         # 1M bars
REV_MAX_AGE = 60       # 1M bars (sweep freshness)
KZ_START, KZ_END = 9 * 60 + 30, 13 * 60   # 09:30 .. 13:00 ET (NY morning session)
                       # the edge is an AM/lunch-liquidity effect: morning-only is the quality
                       # peak (305 trades, 52.8% win, PF 2.15, avgR +0.54, -6R DD) and holds in
                       # every year (2023 PF 2.23 / 2024 2.29 / 2025 1.91). The 13:00-16:00 block
                       # ran at only PF 1.18 / avgR +0.10, so the full 0930-1600 session (493
                       # trades) blends down to PF 1.73 and -8R DD. Set 14*60 for 0930-1400
                       # (more total R, PF ~1.99) or 16*60 to reproduce the old full session.
EOD_MIN = 16 * 60 + 59                     # force-close 16:59 ET
SL_BUF = 2.0
RR = 2.0
MAX_RISK = 55.0
MIN_RISK = 20.0
USE_VOLSTOP = True     # scale the stop floor/cap/buffer by (entry / REF_PRICE) so they stay a
REF_PRICE = 20000.0    # constant % of price (portable across price levels / instruments).
                       # Point values above are the distances at REF_PRICE (NQ 2022-2025 median).
                       # Validated: neutral-to-better on NQ (+167->+176R, PF 2.14->2.25) and it
                       # removes the price-scale distortion that force-closed 56% of trades on the
                       # 2008-2020 OANDA out-of-sample set (see backtest_volstop.py). Set False for
                       # fixed absolute points (original behaviour).
ENTRY_FRAC = 1.0       # 0=enter at 1M FVG near edge (first tap), 1=full fill at far edge.
                       # backtest: 1.0 alone raised win% 43.1->46.4 (PF 1.47->1.66) with MORE
                       # total return from fewer trades; combined with MIT_MAX_AGE=5 above:
                       # ~48% win, PF 1.84, consistent across all 3 years and both 4H anchors.
SWEEP_ALL = True
NEUTRAL_4H_STRUCTURE = True   # neutralMode = "4H structure"
NEUTRAL_LONG_ONLY = True      # neutralLongOnly = true: neutral-bias fallback applies to longs
                              # only (backtest: neutral shorts 29.7% win/PF 0.82 vs neutral
                              # longs 47.3% win/PF 1.67 - NQ trended up over 2022-2025)
DOW_FILTER = False     # OPTIONAL, off by default: restrict entries to Tue/Fri. Backtest:
                       # nominally reaches 51.5% win, but on only 3 years of data with one
                       # year under 50% - weaker, calendar-based edge than the levers above.


def load(csv):
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_datetime(df["timestamp ET"], format="%m/%d/%Y %H:%M")
    out = pd.DataFrame({"o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
                        "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float)},
                       index=pd.DatetimeIndex(t)).sort_index()
    out = out[~out.index.duplicated(keep="first")].dropna()
    return out


def resample(df, rule, origin):
    r = (df.resample(rule, label="left", closed="left", origin=origin)
         .agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna())
    return r


def strict_pivots(h, l, piv):
    """Return (ph, pl): value of a strict pivot high/low CONFIRMED at bar i
    (i.e. the pivot sits at i-piv and is strictly above/below its piv
    neighbours on both sides), else NaN. Matches ta.pivothigh(piv, piv)."""
    n = len(h)
    hs, ls = pd.Series(h), pd.Series(l)
    rmax = hs.rolling(piv).max().to_numpy()   # max of (x-piv+1 .. x)
    rmin = ls.rolling(piv).min().to_numpy()
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(2 * piv, n):
        p = i - piv
        left_hi, right_hi = rmax[p - 1], rmax[i]      # neighbours excl. center
        if h[p] > left_hi and h[p] > right_hi:
            ph[i] = h[p]
        left_lo, right_lo = rmin[p - 1], rmin[i]
        if l[p] < left_lo and l[p] < right_lo:
            pl[i] = l[p]
    return ph, pl


def bias_engine(h, l, c, piv):
    """4H: mitigation -> rejection -> MSS => bias. Per-bar bias/FVG arrays."""
    n = len(c)
    hh = pd.Series(h).rolling(piv * 2 + 1, min_periods=1).max().to_numpy()
    ll = pd.Series(l).rolling(piv * 2 + 1, min_periods=1).min().to_numpy()
    bias = np.zeros(n, int)
    bT = np.full(n, np.nan); bB = np.full(n, np.nan)
    sT = np.full(n, np.nan); sB = np.full(n, np.nan)
    bullTop = bullBot = bearTop = bearBot = np.nan
    bullMit = bullRej = bearMit = bearRej = False
    bullTrig = bearTrig = np.nan
    b = 0
    for i in range(n):
        if i >= 2 and l[i] > h[i - 2]:
            bullTop, bullBot, bullMit, bullRej, bullTrig = l[i], h[i - 2], False, False, np.nan
        if i >= 2 and h[i] < l[i - 2]:
            bearTop, bearBot, bearMit, bearRej, bearTrig = l[i - 2], h[i], False, False, np.nan
        if not np.isnan(bullTop):
            if c[i] < bullBot:
                bullTop = bullBot = np.nan; bullMit = bullRej = False
                if b == 1: b = 0
            else:
                if not bullMit and l[i] <= bullTop:
                    bullMit = True; bullTrig = hh[i - 1] if i > 0 else h[i]
                if bullMit and not bullRej and c[i] > bullTop:
                    bullRej = True
                if bullMit and bullRej and not np.isnan(bullTrig) and c[i] > bullTrig:
                    b = 1
        if not np.isnan(bearTop):
            if c[i] > bearTop:
                bearTop = bearBot = np.nan; bearMit = bearRej = False
                if b == -1: b = 0
            else:
                if not bearMit and h[i] >= bearBot:
                    bearMit = True; bearTrig = ll[i - 1] if i > 0 else l[i]
                if bearMit and not bearRej and c[i] < bearBot:
                    bearRej = True
                if bearMit and bearRej and not np.isnan(bearTrig) and c[i] < bearTrig:
                    b = -1
        bias[i], bT[i], bB[i], sT[i], sB[i] = b, bullTop, bullBot, bearTop, bearBot
    return bias, bT, bB, sT, sB


def flow_engine(h, l, c, piv):
    """15M / 4H: pivot-break order flow + directional FVG mitigation age."""
    n = len(c)
    ph, pl = strict_pivots(h, l, piv)
    flow = np.zeros(n, int)
    bAge = np.full(n, -1, int); sAge = np.full(n, -1, int)
    lastPh = lastPl = np.nan
    f = 0
    bullTop = bullBot = bearTop = bearBot = np.nan
    bMit = sMit = -1
    for i in range(n):
        if not np.isnan(ph[i]): lastPh = ph[i]
        if not np.isnan(pl[i]): lastPl = pl[i]
        if not np.isnan(lastPh) and c[i] > lastPh:
            f = 1; lastPh = np.nan
        if not np.isnan(lastPl) and c[i] < lastPl:
            f = -1; lastPl = np.nan
        if i >= 2 and l[i] > h[i - 2]:
            bullTop, bullBot, bMit = l[i], h[i - 2], -1
        if i >= 2 and h[i] < l[i - 2]:
            bearTop, bearBot, sMit = l[i - 2], h[i], -1
        if not np.isnan(bullTop):
            if c[i] < bullBot:
                bullTop = bullBot = np.nan; bMit = -1
            elif bMit < 0:
                if l[i] <= bullTop: bMit = 0
            else:
                bMit += 1
        if not np.isnan(bearTop):
            if c[i] > bearTop:
                bearTop = bearBot = np.nan; sMit = -1
            elif sMit < 0:
                if h[i] >= bearBot: sMit = 0
            else:
                sMit += 1
        flow[i], bAge[i], sAge[i] = f, bMit, sMit
    return flow, bAge, sAge


def confirmed_index(bar_open_ns, htf_index, bar_span):
    """Last HTF bar fully closed at/before each 1M bar open (no repaint)."""
    ends = (htf_index + bar_span).asi8
    return np.searchsorted(ends, bar_open_ns, side="right") - 1


def run(origin_label):
    df = load(CSV)
    origin = "start_day" if origin_label == "midnight" else \
        pd.Timestamp("1970-01-01 18:00:00")   # 18:00 ET session anchor
    d4 = resample(df, "4h", origin)
    d15 = resample(df, "15min", "start_day")

    b4, b4bT, b4bB, b4sT, b4sB = bias_engine(d4.h.values, d4.l.values, d4.c.values, PIV_BIAS)
    f4, _, _ = flow_engine(d4.h.values, d4.l.values, d4.c.values, PIV_BIAS)
    f15, a15b, a15s = flow_engine(d15.h.values, d15.l.values, d15.c.values, PIV_FLOW)

    t = df.index.asi8
    i4 = confirmed_index(t, d4.index, pd.Timedelta(hours=4))
    i15 = confirmed_index(t, d15.index, pd.Timedelta(minutes=15))

    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    n = len(df)
    mins = (df.index.hour * 60 + df.index.minute).to_numpy()
    day = df.index.normalize().asi8
    dow = df.index.dayofweek.to_numpy()  # Mon=0 .. Fri=4

    phC, plC = strict_pivots(h, l, PIV_CHART)
    prevLo = pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy()
    prevHi = pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy()

    lastPh = lastPl = swingHi = swingLo = np.nan
    mssDir, mssBar = 0, -10**9
    sellSweepBar = buySweepBar = h4BullTouch = h4BearTouch = -10**9
    eBT = eBB = np.nan; eBbar = -1; eBdone = False
    eST = eSB = np.nan; eSbar = -1; eSdone = False

    pos = None
    trades, n_signals = [], 0

    for i in range(n):
        # ---- manage open trade (stop priority) ----
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
                trades.append({**pos, "exit_time": df.index[i], "exit": px,
                               "result": res, "points": pts, "r": pts / pos["risk"]})
                pos = None

        # ---- 1M pivots + MSS ----
        if not np.isnan(phC[i]): lastPh = swingHi = phC[i]
        if not np.isnan(plC[i]): lastPl = swingLo = plC[i]
        if not np.isnan(lastPh) and c[i] > lastPh:
            mssDir, mssBar, lastPh = 1, i, np.nan
        if not np.isnan(lastPl) and c[i] < lastPl:
            mssDir, mssBar, lastPl = -1, i, np.nan
        mssRecent = (i - mssBar) <= MSS_MAX_AGE

        # ---- liquidity sweeps ----
        if not np.isnan(prevLo[i]):
            if l[i] < prevLo[i] and c[i] > prevLo[i]: sellSweepBar = i
            if h[i] > prevHi[i] and c[i] < prevHi[i]: buySweepBar = i

        j4, j15 = i4[i], i15[i]
        if j4 < 0 or j15 < 0:
            continue
        rawBias = b4[j4]
        biasLong = rawBias == 1 or (rawBias == 0 and NEUTRAL_4H_STRUCTURE and f4[j4] == 1)
        biasShort = rawBias == -1 or (rawBias == 0 and NEUTRAL_4H_STRUCTURE and not NEUTRAL_LONG_ONLY and f4[j4] == -1)
        bullT4, bearB4 = b4bT[j4], b4sB[j4]
        flow = f15[j15]
        mitL = 0 <= a15b[j15] <= MIT_MAX_AGE
        mitS = 0 <= a15s[j15] <= MIT_MAX_AGE

        if not np.isnan(bullT4) and l[i] <= bullT4: h4BullTouch = i
        if not np.isnan(bearB4) and h[i] >= bearB4: h4BearTouch = i

        setupLong = (biasLong and flow == 1 and mitL and mssDir == 1 and mssRecent)
        setupShort = (biasShort and flow == -1 and mitS and mssDir == -1 and mssRecent)
        if SWEEP_ALL:
            setupLong = setupLong and (i - sellSweepBar) <= REV_MAX_AGE
            setupShort = setupShort and (i - buySweepBar) <= REV_MAX_AGE

        # ---- 1M entry FVG register / invalidate ----
        if i >= 2:
            if l[i] > h[i - 2] and setupLong:
                eBT, eBB, eBbar, eBdone = l[i], h[i - 2], i, False
            if h[i] < l[i - 2] and setupShort:
                eST, eSB, eSbar, eSdone = l[i - 2], h[i], i, False
        if not np.isnan(eBT) and c[i] < eBB: eBT = eBB = np.nan
        if not np.isnan(eST) and c[i] > eST: eST = eSB = np.nan

        inKZ = KZ_START <= mins[i] < KZ_END
        inDow = (not DOW_FILTER) or dow[i] in (1, 4)  # Tue=1, Fri=4
        flat = pos is None

        # ---- long trigger ----
        if (setupLong and not np.isnan(eBT) and not eBdone and inKZ and inDow
                and i > eBbar):
          trigL = eBT - ENTRY_FRAC * (eBT - eBB)
          if l[i] <= trigL:
            eBdone = True
            entry = trigL
            sc = (entry / REF_PRICE) if USE_VOLSTOP else 1.0
            buf, minr, maxr = SL_BUF * sc, MIN_RISK * sc, MAX_RISK * sc
            base = min(eBB, swingLo) if not np.isnan(swingLo) else eBB
            sl = base - buf
            if minr > 0: sl = min(sl, entry - minr)
            risk = entry - sl
            if not (maxr > 0 and risk > maxr):
                n_signals += 1
                if flat:
                    tp = entry + RR * risk
                    rec = {"dir": 1, "entry_time": df.index[i], "entry": entry,
                           "sl": sl, "tp": tp, "risk": risk,
                           "type": "continuation" if b4[j4] == 1 else "neutral"}
                    if l[i] <= sl:   # trigger bar also spans the stop
                        trades.append({**rec, "exit_time": df.index[i], "exit": sl,
                                       "result": "SL", "points": sl - entry, "r": -1.0})
                    else:
                        pos = rec

        # ---- short trigger ----
        elif (setupShort and not np.isnan(eST) and not eSdone and inKZ and inDow
                and i > eSbar):
          trigS = eSB + ENTRY_FRAC * (eST - eSB)
          if h[i] >= trigS:
            eSdone = True
            entry = trigS
            sc = (entry / REF_PRICE) if USE_VOLSTOP else 1.0
            buf, minr, maxr = SL_BUF * sc, MIN_RISK * sc, MAX_RISK * sc
            base = max(eST, swingHi) if not np.isnan(swingHi) else eST
            sl = base + buf
            if minr > 0: sl = max(sl, entry + minr)
            risk = sl - entry
            if not (maxr > 0 and risk > maxr):
                n_signals += 1
                if flat:
                    tp = entry - RR * risk
                    rec = {"dir": -1, "entry_time": df.index[i], "entry": entry,
                           "sl": sl, "tp": tp, "risk": risk,
                           "type": "continuation" if b4[j4] == -1 else "neutral"}
                    if h[i] >= sl:
                        trades.append({**rec, "exit_time": df.index[i], "exit": sl,
                                       "result": "SL", "points": entry - sl, "r": -1.0})
                    else:
                        pos = rec

    if pos is not None:
        pts = (c[-1] - pos["entry"]) * pos["dir"]
        trades.append({**pos, "exit_time": df.index[-1], "exit": c[-1],
                       "result": "EOD", "points": pts, "r": pts / pos["risk"]})

    tr = pd.DataFrame(trades)
    return df, d4, d15, tr, n_signals


def stats(tr, label):
    n = len(tr)
    if n == 0:
        print(f"{label}: no trades"); return
    wins = (tr.r > 0).sum()
    tot = tr.r.sum()
    pos_r = tr.loc[tr.r > 0, "r"].sum()
    neg_r = -tr.loc[tr.r < 0, "r"].sum()
    pf = pos_r / neg_r if neg_r else float("inf")
    eq = tr.r.cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"{label:<16} trades={n:<4} win%={wins/n*100:5.1f}  totR={tot:+7.1f}  "
          f"avgR={tr.r.mean():+6.3f}  PF={pf:4.2f}  maxDD={dd:6.1f}R  pts={tr.points.sum():+9.1f}")


def report(origin_label):
    df, d4, d15, tr, n_signals = run(origin_label)
    print("=" * 96)
    print(f"4H anchoring: {origin_label}")
    print(f"1M bars {len(df):,}  ({df.index[0]} -> {df.index[-1]})   "
          f"4H bars {len(d4):,}   15M bars {len(d15):,}")
    print(f"signals fired: {n_signals}   trades taken (1 pos at a time): {len(tr)}")
    if tr.empty:
        return tr
    stats(tr, "ALL")
    for y in sorted(tr.entry_time.dt.year.unique()):
        stats(tr[tr.entry_time.dt.year == y], f"  {y}")
    stats(tr[tr.dir == 1], "  Longs")
    stats(tr[tr.dir == -1], "  Shorts")
    print("  exits:", tr.result.value_counts().to_dict(),
          f"| avg risk {tr.risk.mean():.1f}pt (median {tr.risk.median():.1f})")
    # cost sensitivity (points deducted per round trip)
    print("  net of costs:", end=" ")
    for cost in (0.0, 1.0, 2.0):
        net_pts = tr.points.sum() - cost * len(tr)
        net_r = (tr.points - cost).div(tr.risk).sum()
        print(f"[{cost:.0f}pt/rt -> {net_pts:+.0f}pts, {net_r:+.1f}R]", end=" ")
    print()
    return tr


if __name__ == "__main__":
    tr_main = report("midnight")
    print()
    report("18:00-ET")
    if tr_main is not None and not tr_main.empty:
        out = "/Users/manjunathb/Documents/FAB/ICT Indicator/backtest_v2_trades.csv"
        tr_main.assign(cum_r=tr_main.r.cumsum()).to_csv(out, index=False)
        print(f"\ntrade log (midnight anchoring): {out}")
