"""Independent Smart Money Concepts detection from OHLC candles."""


def _valid(candle):
    return isinstance(candle, dict) and all(candle.get(key) is not None for key in ("open", "high", "low", "close"))


def detect_smc(candles, lookback=50):
    """Detect liquidity sweeps, order blocks, and fair value gaps."""
    result = {"liquidity_sweeps": [], "order_blocks": {"bullish": [], "bearish": []}, "fair_value_gaps": {"bullish": [], "bearish": []}}
    try:
        candles = [candle for candle in candles or [] if _valid(candle)][-lookback:]
    except (TypeError, ValueError, KeyError):
        return {**result, "error": "invalid candle data"}
    for index in range(2, len(candles)):
        previous = candles[index - 1]
        current = candles[index]
        older = candles[index - 2]
        if current["low"] < previous["low"] and current["close"] > previous["low"]:
            result["liquidity_sweeps"].append({"type": "SSL", "index": index, "level": current["low"]})
        if current["high"] > previous["high"] and current["close"] < previous["high"]:
            result["liquidity_sweeps"].append({"type": "BSL", "index": index, "level": current["high"]})
        if current["close"] > current["open"] and previous["close"] < previous["open"] and current["close"] > previous["high"]:
            result["order_blocks"]["bullish"].append({"zone": {"low": previous["low"], "high": previous["high"]}, "index": index - 1})
        if current["close"] < current["open"] and previous["close"] > previous["open"] and current["close"] < previous["low"]:
            result["order_blocks"]["bearish"].append({"zone": {"low": previous["low"], "high": previous["high"]}, "index": index - 1})
        if current["low"] > older["high"]:
            result["fair_value_gaps"]["bullish"].append({"zone": {"low": older["high"], "high": current["low"]}, "index": index})
        if current["high"] < older["low"]:
            result["fair_value_gaps"]["bearish"].append({"zone": {"low": current["high"], "high": older["low"]}, "index": index})
    return result