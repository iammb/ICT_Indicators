#!/usr/bin/env python3
"""
Out-of-sample backtest on ~13 years of NAS100 1-min data (OANDA, 2008-2020).

This is the genuinely-unseen-data test. The strategy's levers were all tuned on
NQ futures 2022-2025 (a bull market); here we run the FROZEN shipped config on a
completely different, earlier period that contains the regimes the tuning file
never had: the 2008 GFC, the 2011 and 2015-16 selloffs, the 2018-Q4 bear, and the
Feb-Mar 2020 COVID crash.

Caveats stated up front (this is a proxy, read it as directional not exact):
  * instrument is the NASDAQ-100 CASH INDEX via OANDA CFD, not the NQ future -
    intraday % moves and point scale track closely, but it is not the same series
  * OANDA retail CFD feed: thinner/nosier than CME, especially pre-2012; the
    'volume' column is a tick count, not real contract volume
  * source: github.com/FutureSharks/financial-data (HistData/OANDA mirror)
  * timestamps are UTC in the files -> converted to America/New_York (DST-aware)
    so the 09:30-13:00 ET session and the daily/4H resampling line up

Nothing about the strategy is changed or refit. Engine + trade logic are imported
from the trusted modules; only the data source and the UTC->ET load differ.
"""
import glob
import numpy as np
import pandas as pd
from backtest_v2 import resample, bias_engine, flow_engine, strict_pivots, confirmed_index
from backtest_walkforward import PIV, SWEEP_LEN
from backtest_regime import simulate_regime, daily_regime

DATA_DIR = ("/private/tmp/claude-501/-Users-manjunathb-Documents-FAB-ICT-Indicator-"
            "-claude-worktrees-indicator-strategy-backtest-41dedb/"
            "1d49daff-67a6-4afb-9b01-9257a85ee215/scratchpad/oanda")


def load_oanda(dirpath):
    frames = []
    for f in sorted(glob.glob(f"{dirpath}/oanda-*.csv")):
        try:
            with open(f) as fh:
                if not fh.readline().startswith("time,"):
                    continue                       # skip 404/empty files
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    raw = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(raw["time"], format="%Y-%m-%d %H:%M:%S", utc=True)
    t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)   # UTC -> naive ET
    out = pd.DataFrame({"o": raw["open"].to_numpy(float), "h": raw["high"].to_numpy(float),
                        "l": raw["low"].to_numpy(float), "c": raw["close"].to_numpy(float)},
                       index=pd.DatetimeIndex(t)).sort_index()
    out = out[~out.index.duplicated(keep="first")].dropna()
    return out


def build_P(df):
    """Mirror backtest_walkforward.precompute() but from an arbitrary df (midnight 4H anchor)."""
    d4 = resample(df, "4h", "start_day")
    d15 = resample(df, "15min", "start_day")
    b4, b4bT, b4bB, b4sT, b4sB = bias_engine(d4.h.values, d4.l.values, d4.c.values, PIV)
    f4, _, _ = flow_engine(d4.h.values, d4.l.values, d4.c.values, PIV)
    f15, a15b, a15s = flow_engine(d15.h.values, d15.l.values, d15.c.values, PIV)
    t = df.index.asi8
    i4 = confirmed_index(t, d4.index, pd.Timedelta(hours=4))
    i15 = confirmed_index(t, d15.index, pd.Timedelta(minutes=15))
    valid = (i4 >= 0) & (i15 >= 0)
    j4 = np.where(i4 >= 0, i4, 0); j15 = np.where(i15 >= 0, i15, 0)
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    ph, pl = strict_pivots(h, l, PIV)
    return dict(
        df=df, o=o, h=h, l=l, c=c, n=len(df),
        mins=(df.index.hour * 60 + df.index.minute).to_numpy(),
        year=df.index.year.to_numpy(),
        phC=ph, plC=pl,
        prevLo=pd.Series(l).rolling(SWEEP_LEN).min().shift(1).to_numpy(),
        prevHi=pd.Series(h).rolling(SWEEP_LEN).max().shift(1).to_numpy(),
        valid=valid,
        rawBias=b4[j4], f4=f4[j4], bullT4=b4bT[j4], bearB4=b4sB[j4],
        flow=f15[j15], a15b=a15b[j15], a15s=a15s[j15],
    )


def stats(tr, label):
    if len(tr) == 0:
        print(f"{label:<10} no trades"); return
    r = tr.r.to_numpy()
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min()
    pos = r[r > 0].sum(); neg = -r[r < 0].sum(); pf = pos / neg if neg else float("inf")
    print(f"{label:<10} n={len(tr):<5} win%={(r>0).mean()*100:4.1f}  totR={r.sum():+8.1f}  "
          f"avgR={r.mean():+.3f}  PF={pf:4.2f}  maxDD={dd:+7.1f}R  pts={tr.points.sum():+9.0f}")


def main():
    print("Loading OANDA NAS100 1-min files...")
    df = load_oanda(DATA_DIR)
    print(f"{len(df):,} bars  ({df.index[0]} -> {df.index[-1]})")
    P = build_P(df)
    regimeUp, frac_down, ndays = daily_regime(P)
    print(f"Daily bars {ndays}   regime 'down' (below 200-SMA): {frac_down*100:.1f}% of days\n")

    on = simulate_regime(P, regimeUp, guard=True)    # shipped frozen config (regime guard ON)
    off = simulate_regime(P, regimeUp, guard=False)  # static longs-only, for reference

    print("=" * 96)
    print("FROZEN SHIPPED CONFIG on 2008-2020 NAS100 (morning 09:30-13:00, mit=5, frac=1.0, 2R, regime guard ON)")
    print("=" * 96)
    stats(on, "ALL")
    for y in sorted(on.year.unique()):
        stats(on[on.year == y], f"  {y}")
    print("  ---- direction / type ----")
    stats(on[on.dir == 1], "  Longs"); stats(on[on.dir == -1], "  Shorts")
    for ty in sorted(on.type.unique()):
        stats(on[on.type == ty], f"  {ty}")
    print(f"  exits: {on.result.value_counts().to_dict()}  |  avg risk {on.risk.mean():.1f}pt")
    print("  net of costs (pts/round-trip):", end=" ")
    for cost in (0.0, 1.0, 2.0):
        print(f"[{cost:.0f} -> {(on.points-cost).div(on.risk).sum():+.0f}R]", end=" ")
    print("\n" + "-" * 96)
    print("Reference: same data, regime guard OFF (static longs-only neutral fallback)")
    stats(off, "ALL")
    # highlight the stress years
    print("\nStress-regime years (guard ON vs OFF):")
    for y in [2008, 2011, 2015, 2018, 2020]:
        a = on[on.year == y]; b = off[off.year == y]
        if len(a) or len(b):
            ra = a.r.sum() if len(a) else 0.0; rb = b.r.sum() if len(b) else 0.0
            print(f"  {y}:  guard ON {ra:+6.1f}R ({len(a)}t)   OFF {rb:+6.1f}R ({len(b)}t)")


if __name__ == "__main__":
    main()
