# Data

## `stock_prices.csv`: daily adjusted close prices, 20 large US stocks, 1989-12-29 to 2018-04-11

- **Source:** test resources of PyPortfolioOpt (Robert Andrew Martin),
  `tests/resources/stock_prices.csv`, https://github.com/robertmartin8/PyPortfolioOpt (MIT license).
- **SHA-256:** `1f3bc5da6d4b589a34704be69a1a8cd382f643dbedbefbc73319c3b21d9d9c71`
- 7,126 trading days. Ten stocks have full history and no gaps:
  AAPL, GE, AMD, WMT, BAC, T, XOM, BBY, PFE, JPM. The others start later
  (e.g. AMZN 1997, GOOG 2004, FB 2012, BABA 2014).
- **Survivorship bias:** the list was chosen in 2018 from companies that existed
  then. Backtest results are illustrations, not evidence of a tradable strategy.
- Nothing here is investment advice.
