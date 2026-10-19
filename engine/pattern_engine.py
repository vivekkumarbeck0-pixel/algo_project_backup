"""Deterministic one-, two-, and three-candle pattern detection."""


def _valid(candle):
    return all(candle.get(key) is not None for key in ("open", "high", "low", "close"))


def _body(candle):
    return abs(float(candle["close"]) - float(candle["open"]))


def _range(candle):
    return float(candle["high"]) - float(candle["low"])


def _bullish(candle):
    return float(candle["close"]) > float(candle["open"])


def _bearish(candle):
    return float(candle["close"]) < float(candle["open"])


def detect_patterns(candles):
    """Return pattern names and directional confirmations for latest candles."""
    try:
        candles = [c for c in candles or [] if isinstance(c, dict) and _valid(c)]
        if not candles:
            return {"patterns": [], "bullish": False, "bearish": False}
        names = []
        latest = candles[-1]
        latest_range = _range(latest)
        latest_body = _body(latest)
        if latest_range > 0:
            upper_wick = float(latest["high"]) - max(float(latest["open"]), float(latest["close"]))
            lower_wick = min(float(latest["open"]), float(latest["close"])) - float(latest["low"])
            if lower_wick >= max(latest_body * 2, latest_range * 0.5) and upper_wick <= latest_range * 0.25:
                names.append("hammer" if _bullish(latest) else "pinbar")
            if upper_wick >= max(latest_body * 2, latest_range * 0.5) and lower_wick <= latest_range * 0.25:
                names.append("shooting_star")
        if len(candles) >= 2:
            previous = candles[-2]
            if _bearish(previous) and _bullish(latest) and latest["open"] <= previous["close"] and latest["close"] >= previous["open"]:
                names.append("bullish_engulfing")
            if _bullish(previous) and _bearish(latest) and latest["open"] >= previous["close"] and latest["close"] <= previous["open"]:
                names.append("bearish_engulfing")
            if abs(float(previous["low"]) - float(latest["low"])) <= max(_range(previous), _range(latest)) * 0.1 and _bullish(latest):
                names.append("tweezer_bottom")
            if abs(float(previous["high"]) - float(latest["high"])) <= max(_range(previous), _range(latest)) * 0.1 and _bearish(latest):
                names.append("tweezer_top")
        if len(candles) >= 3:
            first, middle, last = candles[-3:]
            middle_small = _body(middle) <= _body(first) * 0.5
            if _bearish(first) and middle_small and _bullish(last) and last["close"] > (first["open"] + first["close"]) / 2:
                names.append("morning_star")
            if _bullish(first) and middle_small and _bearish(last) and last["close"] < (first["open"] + first["close"]) / 2:
                names.append("evening_star")
            if all(_bullish(c) for c in (first, middle, last)) and first["close"] < middle["close"] < last["close"]:
                names.append("three_white_soldiers")
            if all(_bearish(c) for c in (first, middle, last)) and first["close"] > middle["close"] > last["close"]:
                names.append("three_black_crows")
        bullish_names = {"hammer", "pinbar", "bullish_engulfing", "tweezer_bottom", "morning_star", "three_white_soldiers"}
        bearish_names = {"shooting_star", "bearish_engulfing", "tweezer_top", "evening_star", "three_black_crows"}
        return {"patterns": sorted(set(names)), "bullish": bool(set(names) & bullish_names), "bearish": bool(set(names) & bearish_names)}
    except (TypeError, ValueError, KeyError):
        return {"patterns": [], "bullish": False, "bearish": False, "error": "invalid candle data"}
