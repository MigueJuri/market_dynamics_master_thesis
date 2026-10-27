---
name: lobster-format
description: LOBSTER message/orderbook format, sentinels, price scaling, and bin-averaging pitfalls.
---

# LOBSTER format reference

Applies to LOBSTER message and orderbook files used for SPY LOB analysis.

## Message file columns
- `Time`: seconds since midnight (float, can be > 86,400 for after-hours).
- `Event`: 1=new limit, 2=modification, 3=cancel, 4=execution, 5=hidden execution, 7=trading halt indicator.
- `OrderID`, `Size`, `Price`, `Direction` (-1=sell, 1=buy).

## Orderbook file columns
- Alternating `AskPrice_i`, `AskSize_i`, `BidPrice_i`, `BidSize_i` for levels `i=1..N`.
- Prices are in integer ticks (price = value / 10000 for SPY).

## Sentinel values
- `0` size or price can indicate deleted/cancelled levels at the tail of the book.
- `-9999999999` (or similar large negatives) sometimes marks missing price levels in raw dumps.
- Trading-halt events (Event 7) may repeat sentinel rows; filter by session time and event type.

## Session-time filtering
- Regular session: 09:30:00 to 16:00:00 ET.
- Avoid overnight/auction rows unless the analysis explicitly includes them.
- Use `Time` and date from the filename; combine into a proper datetime index.

## Bin-averaging pitfalls
- Choose bin boundaries on integer time grid, not `np.arange` with float step.
- Use `pd.Grouper` or `np.linspace(start, stop, n_bins+1)` for stable bins.
- Last bin may be incomplete; document whether it is dropped or rescaled.
- Mid-quote and spread inside a bin must be event-time weighted when events are irregular.
