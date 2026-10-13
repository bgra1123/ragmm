"""
Rebalance Guard MCP server (stdio).

Analysis only: it computes worst-case risk and proposes trades, it never connects to a broker or places
orders. Keep it separate from any trading connector.
"""
from __future__ import annotations

import functools
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import service
from .core import RebalanceError

INSTRUCTIONS = """\
Rebalance Guard analyses a long-only portfolio and proposes rebalancing trades. It is ANALYSIS ONLY: it cannot and
must not be used to place orders.

How to use it:
- Holdings are {ticker: market value}. Prices are either "demo" (a bundled 10-stock history for demonstrations) or the
  path of a CSV with a date column and one column of adjusted close prices per ticker.
- Call risk_report first to show the portfolio's worst-case risk (EV@R, CVaR, VaR, drawdown, worst periods).
- Call propose_rebalance with mode "min_evar" (minimize the worst-case loss), "min_vol", or "target" (return to target
  weights with the fewest trades once a tolerance band is breached).
- When you present results: state the lookback window and level, show the trades with the estimated cost, explain
  that min_evar/min_vol optimize risk and ignore expected returns, and mention the stability result (how much of
  the proposed trading is estimation noise). Always include the disclaimer. Do not say the tool predicts returns.
"""

server = MCPServer(name="rebalance-guard", instructions=INSTRUCTIONS)
_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def _guarded(fn):
    """Input problems become ToolError so their message reaches the model; other exceptions stay server-side."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RebalanceError as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


@server.tool(annotations=_READ_ONLY)
@_guarded
def describe_demo_data() -> dict[str, Any]:
    """List the tickers and dates of the bundled demo price history (use prices="demo")."""
    return service.describe_demo_data()


@server.tool(annotations=_READ_ONLY)
@_guarded
def risk_report(holdings: dict[str, float], prices: str = "demo", lookback_days: int = 756,
                horizon_days: int = 21, zeta: float = 0.05) -> dict[str, Any]:
    """
    Worst-case risk report for a portfolio.

    holdings: {ticker: market value}. prices: "demo" or a CSV path (first column date, then one column per ticker).
    Reports annual volatility, VaR / CVaR / EV@R of the loss over horizon_days (default 21 trading days) at the
    worst `zeta` of cases, the worst historical periods, max drawdown, concentration and each holding's share of risk.
    """
    return service.risk_report(holdings, prices, lookback_days, horizon_days, zeta)


@server.tool(annotations=_READ_ONLY)
@_guarded
def propose_rebalance(holdings: dict[str, float], prices: str = "demo", mode: str = "min_evar",
                      target_weights: dict[str, float] | None = None, max_weight: float = 1.0,
                      cost_bps: float = 10.0, turnover_cap: float | None = None, trigger_band: float = 0.05,
                      band_type: str = "absolute", landing_band: float | None = None, zeta: float = 0.05,
                      lookback_days: int = 756, horizon_days: int = 21, mean_weight: float = 0.0,
                      stability_samples: int = 30, seed: int = 0) -> dict[str, Any]:
    """
    Propose rebalancing trades (analysis only; no orders are placed).

    mode "min_evar": minimize the worst-case (EV@R) monthly loss. mode "min_vol": minimize volatility.
    mode "target": give target_weights; if any holding is outside trigger_band (absolute weight, or relative
    if band_type="relative"), trade the least needed to bring every holding within landing_band of its target.
    All modes are long-only with max_weight per holding, proportional trading cost cost_bps, and an optional
    turnover_cap (share of the portfolio traded). For the optimizing modes, stability_samples bootstrap
    resamples show how much of the trading is estimation noise and give a 'bagged' proposal.
    """
    return service.propose_rebalance(holdings, prices, mode, target_weights, max_weight, cost_bps, turnover_cap,
                                     trigger_band, band_type, landing_band, zeta, lookback_days, horizon_days,
                                     mean_weight, stability_samples, seed)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
