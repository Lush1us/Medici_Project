"""
Medici — Indicator Registry

Static metadata for every indicator in the master list.
This is the menu Qwen reads from when scoring.
"""

INDICATORS = [
    # =====================================================================
    # MOMENTUM — TA LIBRARY
    # =====================================================================
    {"name": "RSI_14", "category": "momentum", "description": "Relative Strength Index, 14-period. Measures overbought/oversold on 0-100 scale."},
    {"name": "RSI_7", "category": "momentum", "description": "Relative Strength Index, 7-period. Faster RSI variant, more responsive to recent moves."},
    {"name": "STOCH_14_3_3", "category": "momentum", "description": "Stochastic Oscillator. %K and %D lines measure momentum, similar range to RSI."},
    {"name": "WILLR_14", "category": "momentum", "description": "Williams %R, 14-period. Inverted stochastic, measures overbought/oversold on -100 to 0."},
    {"name": "CCI_20", "category": "momentum", "description": "Commodity Channel Index, 20-period. Measures deviation from statistical mean."},
    {"name": "MFI_14", "category": "volume_momentum", "description": "Money Flow Index, 14-period. Volume-weighted RSI variant."},
    {"name": "ROC_12", "category": "momentum", "description": "Rate of Change, 12-period. Simple percentage price change."},
    {"name": "TRIX_15", "category": "momentum", "description": "Triple Exponential Average, 15-period. Rate of change of triple-smoothed EMA, filters noise."},
    {"name": "AO", "category": "momentum", "description": "Awesome Oscillator. Difference between 5 and 34-period SMA of bar midpoints. Detects momentum shifts without close price."},
    {"name": "KAMA_10", "category": "momentum", "description": "Kaufman Adaptive Moving Average, 10-period. Adapts speed to noise — fast in trends, slow in chop. Superior to fixed EMAs."},
    {"name": "PPO", "category": "momentum", "description": "Percentage Price Oscillator. Normalized MACD — comparable across different price levels."},
    {"name": "PVO", "category": "volume_momentum", "description": "Percentage Volume Oscillator. Normalized volume MACD — detects volume surges/droughts vs recent history."},
    {"name": "STOCHRSI_14", "category": "momentum", "description": "Stochastic RSI. Stochastic applied to RSI values. More sensitive OB/OS detection than raw RSI."},
    {"name": "TSI", "category": "momentum", "description": "True Strength Index. Double-smoothed momentum with low noise. Good for trend confirmation and divergence."},
    {"name": "UO", "category": "momentum", "description": "Ultimate Oscillator. Weighted momentum across 3 timeframes (7/14/28). Reduces false signals."},

    # =====================================================================
    # TREND — TA LIBRARY
    # =====================================================================
    {"name": "MACD_12_26_9", "category": "trend", "description": "Moving Average Convergence Divergence (12/26/9). Signal line crossovers indicate trend changes."},
    {"name": "EMA_12", "category": "trend", "description": "12-period Exponential Moving Average. Component of MACD."},
    {"name": "EMA_26", "category": "trend", "description": "26-period Exponential Moving Average. Component of MACD."},
    {"name": "SMA_50", "category": "trend", "description": "50-period Simple Moving Average. Medium-term trend reference."},
    {"name": "SMA_200", "category": "trend", "description": "200-period Simple Moving Average. Long-term trend reference."},
    {"name": "WMA_20", "category": "trend", "description": "Weighted Moving Average, 20-period. Linearly weighted, between SMA and EMA in responsiveness."},
    {"name": "ADX_14", "category": "trend", "description": "Average Directional Index, 14-period. Measures trend strength 0-100, not direction. +DI/-DI show direction."},
    {"name": "ICHIMOKU", "category": "trend", "description": "Ichimoku Cloud. Multi-component trend system: Tenkan, Kijun, Senkou A/B. Cloud acts as support/resistance."},
    {"name": "PSAR", "category": "trend", "description": "Parabolic SAR. Trailing stop-and-reverse dots above/below price."},
    {"name": "AROON_25", "category": "trend", "description": "Aroon indicator, 25-period. Measures time since highest high / lowest low. Trend timing."},
    {"name": "DPO_20", "category": "trend", "description": "Detrended Price Oscillator, 20-period. Removes trend to isolate price cycles."},
    {"name": "KST", "category": "trend", "description": "Know Sure Thing. Weighted sum of 4 smoothed ROCs at different timeframes. Best at identifying major trend changes."},
    {"name": "MASS_INDEX", "category": "trend", "description": "Mass Index. Detects reversal bulges where high-low range expands then contracts."},
    {"name": "STC", "category": "trend", "description": "Schaff Trend Cycle. MACD through stochastic cycle. Faster than MACD with fewer false signals."},
    {"name": "VORTEX_14", "category": "trend", "description": "Vortex Indicator, 14-period. Positive and negative trend movement. Crossovers signal trend changes."},

    # =====================================================================
    # VOLATILITY — TA LIBRARY
    # =====================================================================
    {"name": "ATR_14", "category": "volatility", "description": "Average True Range, 14-period. Measures price volatility in absolute terms."},
    {"name": "BBANDS_20_2", "category": "volatility", "description": "Bollinger Bands (20/2). Volatility envelope around SMA. Includes %B (position within bands) and bandwidth."},
    {"name": "KELTNER_20_1.5", "category": "volatility", "description": "Keltner Channels (20/1.5). ATR-based envelope around EMA, similar to Bollinger but uses ATR."},
    {"name": "STDDEV_20", "category": "volatility", "description": "Standard Deviation, 20-period. Raw volatility measure, component of Bollinger Bands."},
    {"name": "DONCHIAN_20", "category": "volatility", "description": "Donchian Channel, 20-period. Highest high / lowest low. Pure breakout detection, basis of Turtle Trading."},
    {"name": "ULCER_INDEX_14", "category": "volatility", "description": "Ulcer Index, 14-period. Measures depth and duration of drawdowns. Downside-only risk metric."},

    # =====================================================================
    # VOLUME — TA LIBRARY
    # =====================================================================
    {"name": "OBV", "category": "volume", "description": "On-Balance Volume. Cumulative volume flow confirming price trends."},
    {"name": "CMF_20", "category": "volume", "description": "Chaikin Money Flow, 20-period. Accumulation/distribution based on close position within range."},
    {"name": "VWAP", "category": "volume", "description": "Volume-Weighted Average Price. Institutional fair value benchmark."},
    {"name": "ADI", "category": "volume", "description": "Accumulation/Distribution Index. Like OBV but weights by close position within high-low range. More nuanced."},
    {"name": "EOM_14", "category": "volume", "description": "Ease of Movement, 14-period smoothed. Measures how easily price moves relative to volume."},
    {"name": "FORCE_INDEX_13", "category": "volume", "description": "Force Index, 13-period. Price change * volume. Combines direction, magnitude, and volume."},
    {"name": "NVI", "category": "volume", "description": "Negative Volume Index. Only changes on down-volume days. Theory: smart money trades on quiet days."},
    {"name": "VPT", "category": "volume", "description": "Volume Price Trend. Like OBV but weights by percentage price change. Large moves get more attribution."},

    # =====================================================================
    # VOLATILITY REGIME — CUSTOM
    # =====================================================================
    {"name": "GARMAN_KLASS_20", "category": "vol_regime", "description": "Garman-Klass Volatility, 20-period. OHLC-based estimator, 5-8x more efficient than close-only vol."},
    {"name": "PARKINSON_20", "category": "vol_regime", "description": "Parkinson Volatility, 20-period. Range-based (high-low) vol estimator. ~5x more efficient than close-only."},
    {"name": "ROGERS_SATCHELL_20", "category": "vol_regime", "description": "Rogers-Satchell Volatility, 20-period. Range-based, handles trending markets (drift). Unbiased."},
    {"name": "YANG_ZHANG_20", "category": "vol_regime", "description": "Yang-Zhang Volatility, 20-period. Most efficient OHLC estimator. Handles drift and overnight jumps."},
    {"name": "REALIZED_VOL_10", "category": "vol_regime", "description": "Realized Volatility, 10-day (annualized). Short-term vol from log returns."},
    {"name": "REALIZED_VOL_20", "category": "vol_regime", "description": "Realized Volatility, 20-day (annualized). Standard-term vol from log returns."},
    {"name": "REALIZED_VOL_60", "category": "vol_regime", "description": "Realized Volatility, 60-day (annualized). Medium-term vol from log returns."},
    {"name": "VOL_RATIO_10_60", "category": "vol_regime", "description": "Volatility Ratio (10d/60d). Above 1 = vol expanding (regime shift). Below 1 = vol compressing."},
    {"name": "VOL_PERCENTILE_252", "category": "vol_regime", "description": "Vol Percentile (current 20d vol ranked over 1 year). Shows if current vol is historically high/low for this asset."},
    {"name": "NATR_14", "category": "vol_regime", "description": "Normalized ATR (ATR/Close as %). Comparable across different price levels. Position sizing input."},
    {"name": "BB_WIDTH_PERCENTILE_252", "category": "vol_regime", "description": "BB Width Percentile (1-year rank). Low = squeeze = impending breakout (TTM Squeeze basis)."},

    # =====================================================================
    # MICROSTRUCTURE — CUSTOM
    # =====================================================================
    {"name": "AMIHUD_20", "category": "microstructure", "description": "Amihud Illiquidity, 20-day. Price impact per unit of volume. Higher = less liquid, more risk premium."},
    {"name": "ROLL_SPREAD_20", "category": "microstructure", "description": "Roll Spread Estimator, 20-day. Effective bid-ask spread from serial return covariance."},
    {"name": "CLV", "category": "microstructure", "description": "Close Location Value. Where price closed within its range (-1 to +1). Proxy for buy/sell pressure."},
    {"name": "BAR_RANGE_RATIO", "category": "microstructure", "description": "Bar Range Ratio. Current range vs 20-day avg. High = unusual activity, institutional involvement or news."},
    {"name": "CORWIN_SCHULTZ_SPREAD", "category": "microstructure", "description": "Corwin-Schultz Spread, 20-day avg. Bid-ask spread estimated from consecutive high-low prices."},

    # =====================================================================
    # VOLUME ANALYSIS — CUSTOM
    # =====================================================================
    {"name": "RVOL_20", "category": "volume_analysis", "description": "Relative Volume vs 20-day avg. Above 1.3 = heavy, below 0.7 = light. Detects abnormal activity."},
    {"name": "VROC_14", "category": "volume_analysis", "description": "Volume Rate of Change, 14-period. Detects volume acceleration/deceleration trends."},
    {"name": "UP_DOWN_VOL_RATIO_20", "category": "volume_analysis", "description": "Up/Down Volume Ratio, 20-day. Persistent >1 = accumulation, <1 = distribution. Divergence = warning."},
    {"name": "PRICE_VOL_CORR_20", "category": "volume_analysis", "description": "Price-Volume Correlation, 20-day. Healthy trends: positive. Negative = exhaustion signal."},

    # =====================================================================
    # MOMENTUM DECOMPOSITION — CUSTOM
    # =====================================================================
    {"name": "COPPOCK", "category": "momentum_adv", "description": "Coppock Curve. Long-term buying opportunity detector after major bottoms."},
    {"name": "ELDER_IMPULSE", "category": "momentum_adv", "description": "Elder Impulse System. EMA slope + MACD histogram: +1=bullish, -1=bearish, 0=neutral. Regime filter."},
    {"name": "EFFICIENCY_RATIO_10", "category": "momentum_adv", "description": "Kaufman Efficiency Ratio, 10-period. 1.0=perfectly trending, 0.0=pure chop. Cleaner than ADX."},
    {"name": "MOM_ACCEL_5", "category": "momentum_adv", "description": "Momentum Acceleration, 5-period. Second derivative of price. Detects when momentum is gaining or losing steam."},
    {"name": "FISHER_TRANSFORM_10", "category": "momentum_adv", "description": "Fisher Transform, 10-period. Normalizes price to Gaussian, creates sharp turning point signals."},
    {"name": "FRAMA_20", "category": "momentum_adv", "description": "Fractal Adaptive MA, 20-period. Adapts speed based on fractal dimension. Fastest in trends, slowest in ranges."},
    {"name": "HURST_100", "category": "momentum_adv", "description": "Hurst Exponent, 100-period. H>0.5=trending (momentum works), H<0.5=mean-reverting, H=0.5=random walk."},

    # =====================================================================
    # MEAN REVERSION — CUSTOM
    # =====================================================================
    {"name": "ZSCORE_50", "category": "mean_reversion", "description": "Z-Score vs 50-SMA. Standard deviations from mean. Beyond +/-2 = statistically extreme."},
    {"name": "ZSCORE_200", "category": "mean_reversion", "description": "Z-Score vs 200-SMA. Long-term deviation. Extreme values = major mean-reversion candidates."},
    {"name": "PERCENTILE_RANK_252", "category": "mean_reversion", "description": "Percentile Rank of Close, 252-day. Non-parametric. 5=deeply oversold, 95=deeply overbought."},
    {"name": "DIST_52W_RANGE", "category": "mean_reversion", "description": "Distance in 52-week range (0-1). 0=at yearly low, 1=at yearly high."},
    {"name": "HALF_LIFE_50", "category": "mean_reversion", "description": "Mean Reversion Half-Life vs 50-SMA (Ornstein-Uhlenbeck). Expected days to revert halfway. Short=fast reversion=tradeable."},
    {"name": "RSI_BULLISH_DIV", "category": "mean_reversion", "description": "RSI Bullish Divergence (auto-detected). Price makes new 14d low but RSI doesn't. 1=present, 0=absent."},
    {"name": "RSI_BEARISH_DIV", "category": "mean_reversion", "description": "RSI Bearish Divergence (auto-detected). Price makes new 14d high but RSI doesn't. 1=present, 0=absent."},

    # =====================================================================
    # STATISTICAL / DISTRIBUTIONAL — CUSTOM
    # =====================================================================
    {"name": "SKEWNESS_20", "category": "statistical", "description": "Rolling Skewness, 20-day. Negative=fat left tail (crash risk). Positive=potential large up moves."},
    {"name": "SKEWNESS_60", "category": "statistical", "description": "Rolling Skewness, 60-day. Longer-term distributional asymmetry."},
    {"name": "KURTOSIS_20", "category": "statistical", "description": "Rolling Kurtosis, 20-day. High=extreme moves more likely. Affects position sizing and stop placement."},
    {"name": "KURTOSIS_60", "category": "statistical", "description": "Rolling Kurtosis, 60-day. Longer-term tail fatness."},
    {"name": "AUTOCORR_LAG1_20", "category": "statistical", "description": "Autocorrelation lag-1, 20-day. Positive=momentum regime, negative=mean reversion, near 0=random."},
    {"name": "MAX_DRAWDOWN_60", "category": "statistical", "description": "Maximum Drawdown, 60-day rolling. Largest peak-to-trough decline. Current risk context."},
    {"name": "ENTROPY_20", "category": "statistical", "description": "Shannon Entropy of returns, 20-day. High=unpredictable/uncertain, low=orderly. Regime detection."},
    {"name": "LOG_RETURN", "category": "statistical", "description": "Daily Log Return. Foundation metric for all statistical calculations."},

    # =====================================================================
    # CROSS-ASSET — REQUIRES ADDITIONAL DATA
    # =====================================================================
    {"name": "CORR_SPY_20", "category": "cross_asset", "description": "Rolling Correlation to SPY, 20-day. Detects beta regime changes. Breakdown = structural shift."},
    {"name": "CORR_SPY_60", "category": "cross_asset", "description": "Rolling Correlation to SPY, 60-day. Longer-term co-movement."},
    {"name": "BETA_SPY_60", "category": "cross_asset", "description": "Rolling Beta to SPY, 60-day. Dynamic factor exposure. Changes signal regime shifts."},
    {"name": "VIX_TERM_STRUCTURE", "category": "cross_asset", "description": "VIX/VIX3M ratio. Above 1 = backwardation = fear/hedging. Below 1 = contango = complacency."},
    {"name": "CREDIT_SPREAD_20", "category": "cross_asset", "description": "Credit Spread Proxy (HYG-TLT), 20-day. Widening = risk-off. One of the best early warning signals."},
    {"name": "BREADTH_PCT_ABOVE_50SMA", "category": "breadth", "description": "% of S&P 500 stocks above 50-SMA. Below 20% = deeply oversold market. Above 80% = overheated."},
    {"name": "BREADTH_PCT_ABOVE_200SMA", "category": "breadth", "description": "% of S&P 500 stocks above 200-SMA. Below 20% = major bottoms. Above 80% = strong bull."},
    {"name": "SECTOR_REL_STRENGTH", "category": "cross_asset", "description": "Sector Relative Strength vs SPY, 20-day. Shows rotation: cyclicals up=risk-on, defensives up=risk-off."},
]


def get_registry():
    """Return the full indicator registry."""
    return INDICATORS


def get_registry_for_scoring():
    """Return version suitable for Qwen scoring."""
    return [{"name": i["name"], "category": i["category"], "description": i["description"]} for i in INDICATORS]
