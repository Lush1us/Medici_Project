"""
Medici — Indicator Engine

Computes all technical indicators on a ticker's OHLCV dataframe.
Split into: ta library indicators, custom statistical/regime indicators, cross-asset indicators.
Returns a dict of {indicator_name: value_or_dict} for the latest row.
"""

import pandas as pd
import numpy as np
import ta


def compute_all(df: pd.DataFrame, spy_df: pd.DataFrame = None,
                universe_dfs: dict = None, sector_dfs: dict = None,
                vix3m_df: pd.DataFrame = None, hyg_df: pd.DataFrame = None,
                tlt_df: pd.DataFrame = None) -> dict:
    """
    Compute all indicators on df (expects columns: Open, High, Low, Close, Volume).
    Optional dataframes enable cross-asset indicators.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]
    log_ret = np.log(close / close.shift(1))

    results = {}

    # =========================================================================
    # SECTION 1: TA LIBRARY — MOMENTUM
    # =========================================================================

    results["RSI_14"] = _last(ta.momentum.rsi(close, window=14))
    results["RSI_7"] = _last(ta.momentum.rsi(close, window=7))

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    results["STOCH_14_3_3"] = {"k": _last(stoch.stoch()), "d": _last(stoch.stoch_signal())}

    results["WILLR_14"] = _last(ta.momentum.williams_r(high, low, close, lbp=14))
    results["CCI_20"] = _last(ta.trend.cci(high, low, close, window=20))
    results["MFI_14"] = _last(ta.volume.money_flow_index(high, low, close, volume, window=14))
    results["ROC_12"] = _last(ta.momentum.roc(close, window=12))
    results["TRIX_15"] = _last(ta.trend.trix(close, window=15))

    # New momentum from ta
    results["AO"] = _last(ta.momentum.awesome_oscillator(high, low))

    kama_obj = ta.momentum.KAMAIndicator(close, window=10, pow1=2, pow2=30)
    results["KAMA_10"] = _last(kama_obj.kama())

    ppo_obj = ta.momentum.PercentagePriceOscillator(close, window_slow=26, window_fast=12, window_sign=9)
    results["PPO"] = {"ppo": _last(ppo_obj.ppo()), "signal": _last(ppo_obj.ppo_signal()), "histogram": _last(ppo_obj.ppo_hist())}

    pvo_obj = ta.momentum.PercentageVolumeOscillator(volume, window_slow=26, window_fast=12, window_sign=9)
    results["PVO"] = {"pvo": _last(pvo_obj.pvo()), "signal": _last(pvo_obj.pvo_signal()), "histogram": _last(pvo_obj.pvo_hist())}

    stochrsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    results["STOCHRSI_14"] = {"k": _last(stochrsi.stochrsi_k()), "d": _last(stochrsi.stochrsi_d())}

    tsi = ta.momentum.TSIIndicator(close, window_slow=25, window_fast=13)
    results["TSI"] = _last(tsi.tsi())

    results["UO"] = _last(ta.momentum.ultimate_oscillator(high, low, close))

    # =========================================================================
    # SECTION 2: TA LIBRARY — TREND
    # =========================================================================

    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    results["MACD_12_26_9"] = {
        "macd": _last(macd_obj.macd()),
        "signal": _last(macd_obj.macd_signal()),
        "histogram": _last(macd_obj.macd_diff()),
    }

    results["EMA_12"] = _last(ta.trend.ema_indicator(close, window=12))
    results["EMA_26"] = _last(ta.trend.ema_indicator(close, window=26))
    results["SMA_50"] = _last(ta.trend.sma_indicator(close, window=50))
    results["SMA_200"] = _last(ta.trend.sma_indicator(close, window=200))
    results["WMA_20"] = _last(ta.trend.wma_indicator(close, window=20))

    adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
    results["ADX_14"] = {
        "adx": _last(adx_obj.adx()),
        "plus_di": _last(adx_obj.adx_pos()),
        "minus_di": _last(adx_obj.adx_neg()),
    }

    ichi = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    results["ICHIMOKU"] = {
        "tenkan": _last(ichi.ichimoku_conversion_line()),
        "kijun": _last(ichi.ichimoku_base_line()),
        "senkou_a": _last(ichi.ichimoku_a()),
        "senkou_b": _last(ichi.ichimoku_b()),
    }

    results["PSAR"] = _last(ta.trend.psar_down(high, low, close, step=0.02, max_step=0.2))

    aroon = ta.trend.AroonIndicator(high, low, window=25)
    results["AROON_25"] = {"up": _last(aroon.aroon_up()), "down": _last(aroon.aroon_down())}

    # New trend from ta
    results["DPO_20"] = _last(ta.trend.dpo(close, window=20))

    kst = ta.trend.KSTIndicator(close)
    results["KST"] = {"kst": _last(kst.kst()), "signal": _last(kst.kst_sig())}

    results["MASS_INDEX"] = _last(ta.trend.mass_index(high, low))

    results["STC"] = _last(ta.trend.stc(close))

    vortex = ta.trend.VortexIndicator(high, low, close, window=14)
    results["VORTEX_14"] = {"pos": _last(vortex.vortex_indicator_pos()), "neg": _last(vortex.vortex_indicator_neg())}

    # =========================================================================
    # SECTION 3: TA LIBRARY — VOLATILITY
    # =========================================================================

    results["ATR_14"] = _last(ta.volatility.average_true_range(high, low, close, window=14))

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    results["BBANDS_20_2"] = {
        "upper": _last(bb.bollinger_hband()),
        "middle": _last(bb.bollinger_mavg()),
        "lower": _last(bb.bollinger_lband()),
        "pct_b": _last(bb.bollinger_pband()),
        "width": _last(bb.bollinger_wband()),
    }

    kc = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=20, multiplier=1.5)
    results["KELTNER_20_1.5"] = {
        "upper": _last(kc.keltner_channel_hband()),
        "middle": _last(kc.keltner_channel_mband()),
        "lower": _last(kc.keltner_channel_lband()),
        "width": _last(kc.keltner_channel_wband()),
    }

    results["STDDEV_20"] = _last(close.rolling(window=20).std())

    # New volatility from ta
    dc = ta.volatility.DonchianChannel(high, low, close, window=20)
    results["DONCHIAN_20"] = {
        "upper": _last(dc.donchian_channel_hband()),
        "middle": _last(dc.donchian_channel_mband()),
        "lower": _last(dc.donchian_channel_lband()),
        "width": _last(dc.donchian_channel_wband()),
    }

    results["ULCER_INDEX_14"] = _last(ta.volatility.ulcer_index(close, window=14))

    # =========================================================================
    # SECTION 4: TA LIBRARY — VOLUME
    # =========================================================================

    results["OBV"] = _last(ta.volume.on_balance_volume(close, volume))
    results["CMF_20"] = _last(ta.volume.chaikin_money_flow(high, low, close, volume, window=20))

    typical = (high + low + close) / 3
    results["VWAP"] = _last((typical * volume).cumsum() / volume.cumsum())

    # New volume from ta
    results["ADI"] = _last(ta.volume.acc_dist_index(high, low, close, volume))
    results["EOM_14"] = _last(ta.volume.sma_ease_of_movement(high, low, volume, window=14))
    results["FORCE_INDEX_13"] = _last(ta.volume.force_index(close, volume, window=13))
    results["NVI"] = _last(ta.volume.negative_volume_index(close, volume))
    results["VPT"] = _last(ta.volume.volume_price_trend(close, volume))

    # =========================================================================
    # SECTION 5: CUSTOM — VOLATILITY REGIME
    # =========================================================================

    n = 20  # standard window for vol estimators

    # Garman-Klass Volatility
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(n).mean()
    results["GARMAN_KLASS_20"] = _last(np.sqrt(gk * 252))

    # Parkinson Volatility
    park = (np.log(high / low) ** 2 / (4 * np.log(2))).rolling(n).mean()
    results["PARKINSON_20"] = _last(np.sqrt(park * 252))

    # Rogers-Satchell Volatility
    rs = (np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)).rolling(n).mean()
    results["ROGERS_SATCHELL_20"] = _last(np.sqrt(rs.clip(lower=0) * 252))

    # Yang-Zhang Volatility
    overnight = np.log(open_ / close.shift(1))
    overnight_var = overnight.rolling(n).var()
    intraday = np.log(close / open_)
    intraday_var = intraday.rolling(n).var()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    rs_var = rs.clip(lower=0)
    yz_var = overnight_var + k * intraday_var + (1 - k) * rs_var
    results["YANG_ZHANG_20"] = _last(np.sqrt(yz_var.clip(lower=0) * 252))

    # Realized Volatility (short and long)
    results["REALIZED_VOL_10"] = _last(log_ret.rolling(10).std() * np.sqrt(252))
    results["REALIZED_VOL_20"] = _last(log_ret.rolling(20).std() * np.sqrt(252))
    results["REALIZED_VOL_60"] = _last(log_ret.rolling(60).std() * np.sqrt(252))

    # Volatility Ratio (short/long)
    rv_10 = log_ret.rolling(10).std()
    rv_60 = log_ret.rolling(60).std()
    results["VOL_RATIO_10_60"] = _last(rv_10 / rv_60)

    # Volatility Percentile (current 20d vol ranked over 1 year)
    rv_20 = log_ret.rolling(20).std()
    results["VOL_PERCENTILE_252"] = _last(rv_20.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False))

    # Normalized ATR (ATR / Close, as percentage)
    atr_series = ta.volatility.average_true_range(high, low, close, window=14)
    results["NATR_14"] = _last(atr_series / close * 100)

    # Bollinger Band Width Percentile
    bb_width = bb.bollinger_wband()
    results["BB_WIDTH_PERCENTILE_252"] = _last(bb_width.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False))

    # =========================================================================
    # SECTION 6: CUSTOM — MICROSTRUCTURE
    # =========================================================================

    # Amihud Illiquidity
    dollar_vol = close * volume
    amihud = (log_ret.abs() / dollar_vol).rolling(20).mean()
    results["AMIHUD_20"] = _last(amihud * 1e6)  # scale for readability

    # Roll Spread Estimator
    cov = log_ret.rolling(20).apply(lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan, raw=True)
    roll_spread = 2 * np.sqrt((-cov).clip(lower=0))
    results["ROLL_SPREAD_20"] = _last(roll_spread)

    # Close Location Value
    hl_range = high - low
    clv = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    results["CLV"] = _last(clv)

    # Bar Range Ratio (current bar range / 20-day avg range)
    bar_range = high - low
    avg_range = bar_range.rolling(20).mean()
    results["BAR_RANGE_RATIO"] = _last(bar_range / avg_range)

    # Corwin-Schultz Spread Estimator
    beta_cs = (np.log(high.rolling(2).max() / low.rolling(2).min())) ** 2
    gamma_cs = (np.log(high / low)) ** 2
    alpha_cs = (np.sqrt(2 * beta_cs) - np.sqrt(gamma_cs)) / (3 - 2 * np.sqrt(2))
    cs_spread = 2 * (np.exp(alpha_cs) - 1) / (1 + np.exp(alpha_cs))
    results["CORWIN_SCHULTZ_SPREAD"] = _last(cs_spread.rolling(20).mean())

    # =========================================================================
    # SECTION 7: CUSTOM — VOLUME ANALYSIS
    # =========================================================================

    # Relative Volume (vs 20-day average)
    results["RVOL_20"] = _last(volume / volume.rolling(20).mean())

    # Volume Rate of Change
    results["VROC_14"] = _last(volume.pct_change(14) * 100)

    # Up/Down Volume Ratio
    up_vol = (volume * (close > close.shift(1)).astype(float)).rolling(20).sum()
    down_vol = (volume * (close < close.shift(1)).astype(float)).rolling(20).sum()
    results["UP_DOWN_VOL_RATIO_20"] = _last(up_vol / down_vol.replace(0, np.nan))

    # Price-Volume Divergence (correlation over 20 periods)
    pv_corr = close.pct_change().rolling(20).corr(volume.pct_change())
    results["PRICE_VOL_CORR_20"] = _last(pv_corr)

    # =========================================================================
    # SECTION 8: CUSTOM — MOMENTUM DECOMPOSITION
    # =========================================================================

    # Coppock Curve (adapted for daily: 200 + 150 period ROC, smoothed over 100)
    roc_long = close.pct_change(200) * 100
    roc_short = close.pct_change(150) * 100
    coppock = (roc_long + roc_short).ewm(span=100).mean()
    results["COPPOCK"] = _last(coppock)

    # Elder Impulse System
    ema_13 = close.ewm(span=13).mean()
    macd_hist = macd_obj.macd_diff()
    ema_rising = ema_13 > ema_13.shift(1)
    hist_rising = macd_hist > macd_hist.shift(1)
    impulse = pd.Series(0, index=close.index)
    impulse[ema_rising & hist_rising] = 1    # green / bullish
    impulse[~ema_rising & ~hist_rising] = -1  # red / bearish
    results["ELDER_IMPULSE"] = _last(impulse)

    # Efficiency Ratio (Kaufman)
    direction = (close - close.shift(10)).abs()
    volatility_sum = close.diff().abs().rolling(10).sum()
    er = direction / volatility_sum.replace(0, np.nan)
    results["EFFICIENCY_RATIO_10"] = _last(er)

    # Momentum Acceleration (second derivative of price)
    mom_1 = close.diff(5)
    mom_accel = mom_1.diff(5)
    results["MOM_ACCEL_5"] = _last(mom_accel)

    # Fisher Transform
    hl_mid = (high + low) / 2
    min_low = hl_mid.rolling(10).min()
    max_high = hl_mid.rolling(10).max()
    raw_fish = 2 * ((hl_mid - min_low) / (max_high - min_low).replace(0, np.nan) - 0.5)
    raw_fish = raw_fish.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + raw_fish) / (1 - raw_fish))
    fisher = fisher.ewm(span=5).mean()
    results["FISHER_TRANSFORM_10"] = _last(fisher)

    # Fractal Adaptive Moving Average (FRAMA)
    frama_n = 20
    half = frama_n // 2
    if len(close) >= frama_n:
        hh1 = high.rolling(half).max()
        ll1 = low.rolling(half).min()
        hh2 = high.shift(half).rolling(half).max()
        ll2 = low.shift(half).rolling(half).min()
        hh3 = high.rolling(frama_n).max()
        ll3 = low.rolling(frama_n).min()
        n1 = (hh1 - ll1) / half
        n2 = (hh2 - ll2) / half
        n3 = (hh3 - ll3) / frama_n
        dimen = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        dimen = dimen.replace([np.inf, -np.inf], np.nan)
        alpha_f = np.exp(-4.6 * (dimen - 1)).clip(0.01, 1)
        frama = pd.Series(np.nan, index=close.index)
        frama.iloc[frama_n - 1] = close.iloc[:frama_n].mean()
        for i in range(frama_n, len(close)):
            if pd.notna(alpha_f.iloc[i]) and pd.notna(frama.iloc[i - 1]):
                frama.iloc[i] = alpha_f.iloc[i] * close.iloc[i] + (1 - alpha_f.iloc[i]) * frama.iloc[i - 1]
        results["FRAMA_20"] = _last(frama)
    else:
        results["FRAMA_20"] = None

    # Hurst Exponent (rescaled range, rolling 100 periods)
    results["HURST_100"] = _hurst(log_ret, 100)

    # =========================================================================
    # SECTION 9: CUSTOM — MEAN REVERSION
    # =========================================================================

    # Z-Score of Close vs 50-SMA
    sma_50 = close.rolling(50).mean()
    std_50 = close.rolling(50).std()
    results["ZSCORE_50"] = _last((close - sma_50) / std_50.replace(0, np.nan))

    # Z-Score of Close vs 200-SMA
    sma_200 = close.rolling(200).mean()
    std_200 = close.rolling(200).std()
    results["ZSCORE_200"] = _last((close - sma_200) / std_200.replace(0, np.nan))

    # Percentile Rank of Close (over 252 days)
    results["PERCENTILE_RANK_252"] = _last(close.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False))

    # Distance from 52-week High/Low
    hi_252 = close.rolling(252).max()
    lo_252 = close.rolling(252).min()
    results["DIST_52W_RANGE"] = _last((close - lo_252) / (hi_252 - lo_252).replace(0, np.nan))

    # Mean Reversion Half-Life (Ornstein-Uhlenbeck, on spread from 50-SMA)
    spread = close - sma_50
    if len(spread.dropna()) > 50:
        spread_clean = spread.dropna()
        lag = spread_clean.shift(1).dropna()
        delta = spread_clean.diff().dropna()
        common = lag.index.intersection(delta.index)
        if len(common) > 20:
            lag_v = lag.loc[common].values
            delta_v = delta.loc[common].values
            beta_hl = np.polyfit(lag_v, delta_v, 1)[0]
            if beta_hl < 0:
                results["HALF_LIFE_50"] = round(-np.log(2) / beta_hl, 2)
            else:
                results["HALF_LIFE_50"] = None
        else:
            results["HALF_LIFE_50"] = None
    else:
        results["HALF_LIFE_50"] = None

    # RSI Divergence (automated: price makes new 14d low but RSI doesn't)
    rsi_14 = ta.momentum.rsi(close, window=14)
    price_new_low = close == close.rolling(14).min()
    rsi_at_low = rsi_14[price_new_low]
    if len(rsi_at_low) >= 2:
        results["RSI_BULLISH_DIV"] = 1 if rsi_at_low.iloc[-1] > rsi_at_low.iloc[-2] else 0
    else:
        results["RSI_BULLISH_DIV"] = None
    price_new_high = close == close.rolling(14).max()
    rsi_at_high = rsi_14[price_new_high]
    if len(rsi_at_high) >= 2:
        results["RSI_BEARISH_DIV"] = 1 if rsi_at_high.iloc[-1] < rsi_at_high.iloc[-2] else 0
    else:
        results["RSI_BEARISH_DIV"] = None

    # =========================================================================
    # SECTION 10: CUSTOM — STATISTICAL / DISTRIBUTIONAL
    # =========================================================================

    results["SKEWNESS_20"] = _last(log_ret.rolling(20).skew())
    results["SKEWNESS_60"] = _last(log_ret.rolling(60).skew())
    results["KURTOSIS_20"] = _last(log_ret.rolling(20).kurt())
    results["KURTOSIS_60"] = _last(log_ret.rolling(60).kurt())

    # Autocorrelation of returns (lag 1)
    results["AUTOCORR_LAG1_20"] = _last(log_ret.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False))

    # Maximum Drawdown (rolling 60-day)
    rolling_max = close.rolling(60).max()
    drawdown = (close - rolling_max) / rolling_max
    results["MAX_DRAWDOWN_60"] = _last(drawdown)

    # Shannon Entropy of returns (20-day, binned)
    def _entropy(x):
        counts, _ = np.histogram(x, bins=10)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    results["ENTROPY_20"] = _last(log_ret.rolling(20).apply(_entropy, raw=True))

    # Daily Log Return
    results["LOG_RETURN"] = _last(log_ret)

    # =========================================================================
    # SECTION 11: CROSS-ASSET INDICATORS (require additional dataframes)
    # =========================================================================

    # Rolling Correlation & Beta to SPY
    if spy_df is not None and len(spy_df) > 0:
        spy_ret = np.log(spy_df["Close"] / spy_df["Close"].shift(1))
        common_idx = log_ret.index.intersection(spy_ret.index)
        if len(common_idx) > 60:
            lr = log_ret.reindex(common_idx)
            sr = spy_ret.reindex(common_idx)
            results["CORR_SPY_20"] = _last(lr.rolling(20).corr(sr))
            results["CORR_SPY_60"] = _last(lr.rolling(60).corr(sr))
            cov_spy = lr.rolling(60).cov(sr)
            var_spy = sr.rolling(60).var()
            results["BETA_SPY_60"] = _last(cov_spy / var_spy.replace(0, np.nan))
        else:
            results["CORR_SPY_20"] = None
            results["CORR_SPY_60"] = None
            results["BETA_SPY_60"] = None
    else:
        results["CORR_SPY_20"] = None
        results["CORR_SPY_60"] = None
        results["BETA_SPY_60"] = None

    # VIX Term Structure (VIX / VIX3M ratio — backwardation when > 1)
    if vix3m_df is not None and len(vix3m_df) > 0:
        vix_close = df.get("VIX_Close")  # would need to be joined upstream
        # Handled at pipeline level since VIX data is separate
        results["VIX_TERM_STRUCTURE"] = None  # placeholder, computed in pipeline
    else:
        results["VIX_TERM_STRUCTURE"] = None

    # Credit Spread Proxy (HYG - TLT return divergence)
    if hyg_df is not None and tlt_df is not None:
        hyg_ret = np.log(hyg_df["Close"] / hyg_df["Close"].shift(1))
        tlt_ret = np.log(tlt_df["Close"] / tlt_df["Close"].shift(1))
        common_idx = hyg_ret.index.intersection(tlt_ret.index)
        if len(common_idx) > 20:
            spread = hyg_ret.reindex(common_idx) - tlt_ret.reindex(common_idx)
            results["CREDIT_SPREAD_20"] = _last(spread.rolling(20).mean() * 252 * 100)
        else:
            results["CREDIT_SPREAD_20"] = None
    else:
        results["CREDIT_SPREAD_20"] = None

    # Breadth: % of universe above 50-SMA and 200-SMA
    if universe_dfs:
        above_50 = 0
        above_200 = 0
        total = 0
        for sym, udf in universe_dfs.items():
            if len(udf) < 200:
                continue
            c = udf["Close"]
            total += 1
            if c.iloc[-1] > c.rolling(50).mean().iloc[-1]:
                above_50 += 1
            if c.iloc[-1] > c.rolling(200).mean().iloc[-1]:
                above_200 += 1
        if total > 0:
            results["BREADTH_PCT_ABOVE_50SMA"] = round(above_50 / total * 100, 2)
            results["BREADTH_PCT_ABOVE_200SMA"] = round(above_200 / total * 100, 2)
        else:
            results["BREADTH_PCT_ABOVE_50SMA"] = None
            results["BREADTH_PCT_ABOVE_200SMA"] = None
    else:
        results["BREADTH_PCT_ABOVE_50SMA"] = None
        results["BREADTH_PCT_ABOVE_200SMA"] = None

    # Sector Relative Strength (if sector ETFs provided)
    if sector_dfs:
        sector_rs = {}
        for sec, sdf in sector_dfs.items():
            if spy_df is not None and len(sdf) > 20 and len(spy_df) > 20:
                sec_ret = sdf["Close"].pct_change(20).iloc[-1]
                spy_ret_20 = spy_df["Close"].pct_change(20).iloc[-1]
                if pd.notna(sec_ret) and pd.notna(spy_ret_20) and spy_ret_20 != 0:
                    sector_rs[sec] = round((sec_ret - spy_ret_20) * 100, 2)
        if sector_rs:
            results["SECTOR_REL_STRENGTH"] = sector_rs
        else:
            results["SECTOR_REL_STRENGTH"] = None
    else:
        results["SECTOR_REL_STRENGTH"] = None

    return results


# =========================================================================
# HELPERS
# =========================================================================

def _last(series) -> float | None:
    """Extract last non-NaN value from a series, rounded."""
    if series is None:
        return None
    try:
        val = series.iloc[-1]
    except (IndexError, AttributeError):
        return None
    if pd.isna(val):
        return None
    return round(float(val), 4)


def _hurst(log_returns, window):
    """Rolling Hurst exponent via rescaled range method."""
    if log_returns is None or len(log_returns.dropna()) < window:
        return None
    data = log_returns.dropna().iloc[-window:]
    n = len(data)
    if n < 20:
        return None

    max_k = min(n // 2, 50)
    sizes = []
    rs_values = []

    for k in range(10, max_k + 1, 5):
        rs_list = []
        for start in range(0, n - k + 1, k):
            chunk = data.iloc[start:start + k].values
            mean_c = chunk.mean()
            deviate = np.cumsum(chunk - mean_c)
            r = deviate.max() - deviate.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            sizes.append(k)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 3:
        return None

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    try:
        h = np.polyfit(log_sizes, log_rs, 1)[0]
        return round(float(h), 4)
    except (np.linalg.LinAlgError, ValueError):
        return None
