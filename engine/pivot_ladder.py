"""Exact pivot ladder and current-price support/resistance selection."""


def classic_pivots(high, low, close):
    pp = (float(high) + float(low) + float(close)) / 3.0
    spread = float(high) - float(low)
    return {"pp": pp, "s1": 2 * pp - float(high), "r1": 2 * pp - float(low), "s2": pp - spread, "r2": pp + spread}


def select_current_sr(pivots, current_price):
    levels = sorted({float(value) for value in pivots.values() if value is not None})
    if not levels or current_price is None:
        return None, None
    below = [level for level in levels if level <= float(current_price)]
    above = [level for level in levels if level >= float(current_price)]
    return (max(below) if below else levels[0], min(above) if above else levels[-1])


def _positive_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _oi_peak_strikes(option_chain):
    grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
    peaks = set()
    for raw_strike, sides in grouped.items():
        strike = _number(raw_strike)
        if strike is None:
            continue
        for option_type in ("CE", "PE"):
            quote = ((sides or {}).get(option_type, {}) or {})
            for field in ("open_interest", "oi_change"):
                value = _positive_number(quote.get(field))
                if value is not None:
                    peaks.add((option_type, field, value, strike))
    winners = set()
    for option_type in ("CE", "PE"):
        for field in ("open_interest", "oi_change"):
            rows = [row for row in peaks if row[0] == option_type and row[1] == field]
            if rows:
                peak_value = max(row[2] for row in rows)
                winners.update(row[3] for row in rows if row[2] == peak_value)
    return winners


def _directional_oi_cutoff(option_chain, spot, option_type):
    grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
    metric_rows = {"open_interest": [], "oi_change": []}
    for raw_strike, sides in grouped.items():
        strike = _number(raw_strike)
        if strike is None:
            continue
        if option_type == "CE" and strike <= spot:
            continue
        if option_type == "PE" and strike >= spot:
            continue
        quote = ((sides or {}).get(option_type, {}) or {})
        for metric in metric_rows:
            value = _positive_number(quote.get(metric))
            if value is not None:
                metric_rows[metric].append((value, strike))
    peaks = [
        strike
        for rows in metric_rows.values() if rows
        for value, strike in rows if value == max(item[0] for item in rows)
    ]
    if not peaks:
        return None
    return max(peaks) if option_type == "CE" else min(peaks)


def _number(value):
    try:
        return float(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _candle_sr(candles, reference_price, support_anchor=None, min_gap=47.0, max_gap=53.0):
    valid = []
    for candle in candles or []:
        if not isinstance(candle, dict):
            continue
        high = _number(candle.get("high"))
        low = _number(candle.get("low"))
        if high is not None and low is not None:
            valid.append({"high": high, "low": low})
    valid = valid[-60:]
    if not valid:
        return None, None

    swing_lows = [
        valid[index]["low"]
        for index in range(1, len(valid) - 1)
        if valid[index]["low"] < valid[index - 1]["low"]
        and valid[index]["low"] < valid[index + 1]["low"]
    ]
    swing_highs = [
        valid[index]["high"]
        for index in range(1, len(valid) - 1)
        if valid[index]["high"] > valid[index - 1]["high"]
        and valid[index]["high"] > valid[index + 1]["high"]
    ]

    support_candidates = [level for level in swing_lows if level <= reference_price]
    if not support_candidates:
        support_candidates = [candle["low"] for candle in valid if candle["low"] <= reference_price]
    resistance_candidates = [level for level in swing_highs if level >= reference_price]
    if not resistance_candidates:
        resistance_candidates = [candle["high"] for candle in valid if candle["high"] >= reference_price]

    if support_anchor is not None and support_anchor <= reference_price:
        valid_resistances = [
            level for level in resistance_candidates
            if min_gap <= level - support_anchor <= max_gap
        ]
        return support_anchor, min(valid_resistances) if valid_resistances else None

    valid_pairs = [
        (support, resistance)
        for support in support_candidates
        for resistance in resistance_candidates
        if min_gap <= resistance - support <= max_gap
    ]
    if not valid_pairs:
        return None, None
    return min(
        valid_pairs,
        key=lambda pair: (
            abs((pair[1] - pair[0]) - 50.0),
            abs(((pair[0] + pair[1]) / 2.0) - reference_price),
        ),
    )


def build_connected_ladder(strikes, pivots, option_chain=None, spot=None, candles=None):
    """Calculate each strike's candle swing S/R, bounded by directional OI peaks."""
    ordered = sorted({float(strike) for strike in strikes})
    levels = sorted({float(value) for value in pivots.values() if value is not None})
    quote_by_strike = {}
    grouped_quotes = (option_chain or {}).get("by_strike", {}) or {}
    for raw_strike, sides in grouped_quotes.items():
        strike = _number(raw_strike)
        if strike is not None:
            quote_by_strike[strike] = sides or {}
    if option_chain is not None and spot is not None:
        spot = float(spot)
        if candles is None:
            ce_cutoff = _directional_oi_cutoff(option_chain, spot, "CE")
            pe_cutoff = _directional_oi_cutoff(option_chain, spot, "PE")
            ordered = [
                strike for strike in ordered
                if (ce_cutoff is not None and spot < strike <= ce_cutoff)
                or (pe_cutoff is not None and pe_cutoff <= strike < spot)
                or (strike == spot and (ce_cutoff is not None or pe_cutoff is not None))
            ]
        else:
            peak_strikes = _oi_peak_strikes(option_chain)
            if peak_strikes:
                lower_bound = min({spot, *peak_strikes})
                upper_bound = max({spot, *peak_strikes})
                ordered = [strike for strike in ordered if lower_bound <= strike <= upper_bound]
            else:
                ordered = []
    ladder = {}
    previous_resistance = None
    for strike in ordered:
        if candles is not None:
            sides = quote_by_strike.get(strike, {})
            ce = sides.get("CE", {}) or {}
            pe = sides.get("PE", {}) or {}
            ce_oi = _number(ce.get("open_interest"))
            ce_change = _number(ce.get("oi_change"))
            pe_oi = _number(pe.get("open_interest"))
            pe_change = _number(pe.get("oi_change"))
            has_strike_oi_data = any(
                value is not None and value > 0
                for value in (ce_oi, ce_change, pe_oi, pe_change)
            )
            support_anchor = previous_resistance if previous_resistance is not None else None
            support, resistance = _candle_sr(candles, strike, support_anchor=support_anchor)
            resistance = resistance if has_strike_oi_data else None
            ladder[strike] = {
                "support": support if has_strike_oi_data else None,
                "resistance": resistance,
                "support_score": max(pe_oi or 0.0, 0.0) + max(pe_change or 0.0, 0.0),
                "resistance_score": max(ce_oi or 0.0, 0.0) + max(ce_change or 0.0, 0.0),
            }
            if resistance is not None:
                previous_resistance = resistance
            continue
        strike_levels = levels
        if option_chain is not None and spot is not None:
            if strike < spot:
                strike_levels = [level for level in levels if pe_cutoff <= level < spot] if pe_cutoff is not None else []
            else:
                strike_levels = [level for level in levels if spot < level <= ce_cutoff] if ce_cutoff is not None else []
        support_candidates = [level for level in strike_levels if level < strike]
        resistance_candidates = [level for level in strike_levels if level > strike]
        ladder[strike] = {
            "support": max(support_candidates) if support_candidates else None,
            "resistance": min(resistance_candidates) if resistance_candidates else None,
        }
    return ladder