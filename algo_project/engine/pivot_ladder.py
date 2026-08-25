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


def build_connected_ladder(strikes, pivots):
    """Build a linked, exact ladder without rounding strike-adjacent levels.

    Each strike keeps the previous strike's resistance as its support anchor;
    its resistance is the next available pivot above that anchor. This makes
    role reversal explicit while preserving raw pivot decimals.
    """
    ordered = sorted({float(strike) for strike in strikes})
    levels = sorted({float(value) for value in pivots.values() if value is not None})
    ladder = {}
    previous_resistance = None
    for strike in ordered:
        support_candidates = [level for level in levels if level <= strike]
        resistance_candidates = [level for level in levels if level >= strike]
        support = previous_resistance if previous_resistance is not None else (max(support_candidates) if support_candidates else None)
        resistance = min(resistance_candidates) if resistance_candidates else (levels[-1] if levels else None)
        ladder[strike] = {"support": support, "resistance": resistance}
        previous_resistance = resistance
    return ladder