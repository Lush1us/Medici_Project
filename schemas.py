"""
Medici — Strict output schemas for all agents and offices.

Every LLM call in the system must conform to one of these schemas.
No freeform text. No markdown. Structured data only.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class Regime(str, Enum):
    RISK_ON = "risk-on"
    RISK_OFF = "risk-off"
    TRANSITIONAL = "transitional"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_ALL = "sell_all"
    HOLD = "hold"
    SHORT = "short"
    COVER = "cover"


class Conviction(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# DEPARTMENT REPORTS (submitted to Capo)
# =============================================================================

class TechnicalReport(BaseModel):
    """Output from Dpt of Technical Analysis."""
    ticker: str
    date: str
    trend: str = Field(description="Bullish / Bearish / Neutral")
    momentum: float = Field(description="Momentum score -10 to 10")
    key_levels: dict = Field(description="support, resistance, pivot")
    signals: list[str] = Field(description="Active signals (e.g. 'MACD bullish cross')")
    conviction: Conviction
    summary: str = Field(max_length=200)


class RiskReport(BaseModel):
    """Output from Dpt of Risk Management."""
    ticker: str
    date: str
    current_exposure_pct: float
    max_position_size_usd: float
    stop_loss_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    correlation_to_portfolio: Optional[float] = None
    flags: list[str] = Field(description="Risk flags (e.g. 'earnings_in_3d', 'high_beta')")
    summary: str = Field(max_length=200)


class SentimentReport(BaseModel):
    """Output from Dpt of Sentiment Analysis."""
    ticker: str
    date: str
    sentiment_score: float = Field(description="-10 (extreme fear) to 10 (extreme greed)")
    catalysts: list[str] = Field(description="Upcoming or recent catalysts")
    flow_signals: list[str] = Field(description="Options flow, insider activity")
    summary: str = Field(max_length=200)


# =============================================================================
# MACRO (Capo's own analysis)
# =============================================================================

class MacroAnalysis(BaseModel):
    """Capo's macro regime assessment."""
    date: str
    regime: Regime
    confidence: int = Field(ge=1, le=10)
    vix_read: str
    breadth_read: str
    credit_read: str
    reasoning: str = Field(max_length=300)
    sectors_favored: list[str]
    sectors_avoid: list[str]


# =============================================================================
# CAPO FINAL REPORT
# =============================================================================

class TradeDirective(BaseModel):
    """Single trade instruction from the Capo."""
    ticker: str
    action: Action
    amount_usd: float
    conviction: Conviction
    reason: str = Field(max_length=150)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class CapoReport(BaseModel):
    """Final output from the Capo — the decision document."""
    date: str
    capo: str
    macro: MacroAnalysis
    portfolio_value: float
    invested_pct: float
    directives: list[TradeDirective]
    rationale: str = Field(max_length=500)
    risk_notes: str = Field(max_length=300)


# =============================================================================
# TICKER DEEP DIVE (on-demand)
# =============================================================================

class TickerDeepDive(BaseModel):
    """Combined analysis when Capo requests a specific ticker review."""
    ticker: str
    date: str
    technical: TechnicalReport
    risk: RiskReport
    sentiment: SentimentReport
    capo_verdict: TradeDirective
