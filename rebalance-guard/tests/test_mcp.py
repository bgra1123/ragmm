import asyncio
import sys
import pathlib

import pytest

mcp_client = pytest.importorskip("mcp.client.client")
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from rebalance_guard.server import server  # noqa: E402

H = {"AAPL": 50000, "XOM": 30000, "JPM": 20000}
ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def test_tools_are_exposed_read_only_and_analysis_only():
    async def go():
        async with mcp_client.Client(server) as c:
            return await c.list_tools(), c.instructions
    tools, instructions = run(go())
    names = {t.name for t in tools.tools}
    assert names == {"describe_demo_data", "risk_report", "propose_rebalance"}
    assert not any(w in n for n in names for w in ("order", "execute", "place", "buy", "sell", "submit"))
    for t in tools.tools:
        assert t.annotations.read_only_hint is True and t.annotations.destructive_hint is False
        assert t.annotations.open_world_hint is False
    assert "ANALYSIS ONLY" in instructions


def test_calls_return_structured_results_and_model_readable_errors():
    async def go():
        async with mcp_client.Client(server) as c:
            ok = await c.call_tool("risk_report", {"holdings": H})
            plan = await c.call_tool("propose_rebalance", {"holdings": H, "stability_samples": 4, "mode": "min_evar"})
            bad = await c.call_tool("risk_report", {"holdings": {"AAPL": 1, "NOPE": 1}})
            bad_arg = await c.call_tool("propose_rebalance", {"holdings": H, "zeta": 0.9})
            return ok, plan, bad, bad_arg
    ok, plan, bad, bad_arg = run(go())
    assert not ok.is_error and ok.structured_content["risk"]["annual_volatility"] > 0
    assert not plan.is_error and plan.structured_content["recommended"] == "point_estimate"
    assert bad.is_error and "no price history for: ['NOPE']" in bad.content[0].text        # message reaches the model
    assert bad_arg.is_error and "zeta" in bad_arg.content[0].text


def test_server_works_over_real_stdio():
    """Spawn the server as a subprocess exactly as Claude would; any stray stdout would break the handshake."""
    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "rebalance_guard"], cwd=str(ROOT))
        async with mcp_client.Client(params) as c:
            return await c.call_tool("describe_demo_data", {}), await c.call_tool("risk_report", {"holdings": H})
    demo, report = run(go())
    assert not demo.is_error and len(demo.structured_content["tickers"]) == 10
    assert not report.is_error and report.structured_content["risk"]["monthly_evar"] > 0
