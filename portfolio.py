"""
Medici — Portfolio Engine

Tracks cash, positions, trades, and P&L for backtesting.
Supports fractional shares.
"""

import json
import os
from datetime import datetime


class Portfolio:
    def __init__(self, starting_cash=1000.0, log_dir=None):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions = {}  # {ticker: {"shares": float, "avg_cost": float}}
        self.trade_log = []
        self.daily_snapshots = []
        self.log_dir = log_dir

    def buy(self, ticker, shares, price, date):
        """Buy shares at given price. Returns True if executed."""
        cost = shares * price
        if cost > self.cash:
            # Buy what we can afford
            shares = self.cash / price
            cost = shares * price
        if shares <= 0:
            return False

        self.cash -= cost

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos["shares"] + shares
            pos["avg_cost"] = (pos["avg_cost"] * pos["shares"] + cost) / total_shares
            pos["shares"] = total_shares
        else:
            self.positions[ticker] = {"shares": shares, "avg_cost": price}

        self.trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": "buy",
            "shares": round(shares, 6),
            "price": round(price, 4),
            "value": round(cost, 4),
            "cash_after": round(self.cash, 4),
        })
        return True

    def sell(self, ticker, shares, price, date):
        """Sell shares at given price. Returns True if executed."""
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]
        shares = min(shares, pos["shares"])  # can't sell more than we have
        if shares <= 0:
            return False

        revenue = shares * price
        self.cash += revenue
        pos["shares"] -= shares

        realized_pnl = (price - pos["avg_cost"]) * shares

        if pos["shares"] < 0.0001:  # effectively zero
            del self.positions[ticker]

        self.trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": "sell",
            "shares": round(shares, 6),
            "price": round(price, 4),
            "value": round(revenue, 4),
            "realized_pnl": round(realized_pnl, 4),
            "cash_after": round(self.cash, 4),
        })
        return True

    def sell_all(self, ticker, price, date):
        """Sell entire position in a ticker."""
        if ticker not in self.positions:
            return False
        return self.sell(ticker, self.positions[ticker]["shares"], price, date)

    def get_total_value(self, prices: dict) -> float:
        """Total portfolio value given current prices {ticker: price}."""
        total = self.cash
        for ticker, pos in self.positions.items():
            p = prices.get(ticker, pos["avg_cost"])
            total += pos["shares"] * p
        return total

    def snapshot(self, date, prices: dict):
        """Take a daily snapshot of portfolio state."""
        total = self.get_total_value(prices)
        positions_detail = {}
        for ticker, pos in self.positions.items():
            p = prices.get(ticker, pos["avg_cost"])
            mv = pos["shares"] * p
            pnl = (p - pos["avg_cost"]) * pos["shares"]
            positions_detail[ticker] = {
                "shares": round(pos["shares"], 6),
                "avg_cost": round(pos["avg_cost"], 4),
                "current_price": round(p, 4),
                "market_value": round(mv, 4),
                "unrealized_pnl": round(pnl, 4),
            }

        snap = {
            "date": date,
            "cash": round(self.cash, 4),
            "positions": positions_detail,
            "total_value": round(total, 4),
            "total_return_pct": round((total / self.starting_cash - 1) * 100, 4),
            "num_positions": len(self.positions),
        }
        self.daily_snapshots.append(snap)
        return snap

    def get_state_for_prompt(self, prices: dict) -> str:
        """Return a concise portfolio state string for the Capo prompt."""
        total = self.get_total_value(prices)
        ret_pct = (total / self.starting_cash - 1) * 100

        lines = [
            f"Cash: ${self.cash:.2f}",
            f"Total Value: ${total:.2f} ({ret_pct:+.2f}%)",
            f"Open Positions: {len(self.positions)}",
        ]

        for ticker, pos in self.positions.items():
            p = prices.get(ticker, pos["avg_cost"])
            mv = pos["shares"] * p
            pnl_pct = (p / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
            lines.append(f"  {ticker}: {pos['shares']:.4f} shares @ ${pos['avg_cost']:.2f} → ${p:.2f} ({pnl_pct:+.1f}%) = ${mv:.2f}")

        recent = self.trade_log[-5:] if self.trade_log else []
        if recent:
            lines.append(f"Last {len(recent)} trades:")
            for t in recent:
                lines.append(f"  {t['date']} {t['action'].upper()} {t['shares']:.4f} {t['ticker']} @ ${t['price']:.2f}")

        return "\n".join(lines)

    def save(self, path=None):
        """Save portfolio state to JSON."""
        if path is None and self.log_dir:
            path = os.path.join(self.log_dir, "portfolio.json")
        if path is None:
            return

        data = {
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "positions": self.positions,
            "trade_log": self.trade_log,
            "daily_snapshots": self.daily_snapshots,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
