"""Probe broker-backed Crude option-chain data and print per-strike SR inputs.

This is intentionally a probe, not bot integration. It fetches FULL market data
from Angel One for the nearest CRUDEOIL option expiry, then reports the raw CE/PE
OI and quote fields needed to validate an exact SR formula before trading uses it.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from angel_one.instrument_reader import InstrumentReader
from angel_one.login import AngelOneLogin
from config import SYMBOL_REGISTRY


def _expiry_key(item):
    try:
        return datetime.strptime(str(item.get("expiry")), "%d%b%Y")
    except (TypeError, ValueError):
        return datetime.max


def _crude_options(reader, expiry=None, center=None, radius=10):
    cfg = SYMBOL_REGISTRY["CRUDEOIL"]
    items = [
        item for item in reader.instruments
        if str(item.get("name", "")).upper() == "CRUDEOIL"
        and str(item.get("exch_seg", "")).upper() == cfg["exchange"]
        and str(item.get("instrumenttype", "")).upper() == cfg["option_instrumenttype"]
        and str(item.get("symbol", "")).upper().endswith(("CE", "PE"))
    ]
    items = [item for item in items if _expiry_key(item) >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]
    if expiry:
        items = [item for item in items if str(item.get("expiry", "")).upper() == expiry.upper()]
    elif items:
        nearest = min(items, key=_expiry_key).get("expiry")
        items = [item for item in items if item.get("expiry") == nearest]
    items.sort(key=lambda item: (float(item.get("strike", 0) or 0), item.get("symbol", "")))
    if center is not None:
        strikes = sorted({float(item.get("strike", 0) or 0) / 100.0 for item in items})
        nearby = sorted(strikes, key=lambda value: abs(value - center))[: max(1, radius * 2 + 1)]
        nearby_set = {round(value, 6) for value in nearby}
        items = [
            item for item in items
            if round(float(item.get("strike", 0) or 0) / 100.0, 6) in nearby_set
        ]
    return items


def _market_rows(response):
    data = response.get("data", {}) if isinstance(response, dict) else {}
    fetched = data.get("fetched", []) if isinstance(data, dict) else data
    return fetched if isinstance(fetched, list) else []


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Probe CRUDEOIL broker option-chain OI/depth")
    parser.add_argument("--expiry", help="Expiry such as 19SEP2026; default is nearest future expiry")
    parser.add_argument("--center", type=float, help="Approximate Crude price; defaults to broker future LTP")
    parser.add_argument("--radius", type=int, default=10, help="Number of nearby strikes on each side")
    parser.add_argument("--output", default="data/crude_option_chain_probe.json")
    args = parser.parse_args()

    reader = InstrumentReader()
    reader.load()
    cfg = SYMBOL_REGISTRY["CRUDEOIL"]
    underlying = reader.resolve_underlying("CRUDEOIL")
    center = args.center
    if center is None and underlying:
        client_for_spot = AngelOneLogin.connect_from_env()
        spot_response = client_for_spot.get_ltp(
            cfg["underlying_exchange"], underlying.get("symbol"), underlying.get("token")
        )
        spot_data = spot_response.get("data", {}) if isinstance(spot_response, dict) else {}
        center = _number(spot_data.get("ltp"))
    options = _crude_options(reader, args.expiry, center=center, radius=args.radius)
    if not options:
        raise SystemExit("No CRUDEOIL option contracts found for the requested expiry")

    client = AngelOneLogin.connect_from_env()
    tokens = [str(item.get("token")) for item in options if item.get("token")]
    response = client.get_market_data(
        "FULL",
        {SYMBOL_REGISTRY["CRUDEOIL"]["exchange"]: tokens},
    )
    rows = _market_rows(response)
    by_token = {str(row.get("symbolToken")): row for row in rows if isinstance(row, dict)}

    result = []
    grouped = {}
    for item in options:
        token = str(item.get("token"))
        quote = by_token.get(token, {})
        strike = _number(item.get("strike"))
        # Angel One stores the instrument-master strike multiplied by 100.
        if strike is not None:
            strike /= 100.0
        right = "PE" if str(item.get("symbol", "")).upper().endswith("PE") else "CE"
        record = {
            "strike": strike,
            "right": right,
            "symbol": item.get("symbol"),
            "token": token,
            "expiry": item.get("expiry"),
            "ltp": quote.get("ltp"),
            "open_interest": quote.get("opnInterest", quote.get("openInterest")),
            "trade_volume": quote.get("tradeVolume"),
            "depth": quote.get("depth"),
            "raw_quote": quote,
        }
        result.append(record)
        grouped.setdefault(str(strike), {})[right] = record

    # Transparent standard OI interpretation: strongest PE OI at/below a
    # reference strike is support; strongest CE OI at/above it is resistance.
    # These are candidates for review only, not bot inputs.
    strike_values = sorted(float(value) for value in grouped)
    oi_candidates = {}
    for reference in strike_values:
        support_rows = [
            grouped[str(strike)].get("PE", {})
            for strike in strike_values if strike <= reference
        ]
        resistance_rows = [
            grouped[str(strike)].get("CE", {})
            for strike in strike_values if strike >= reference
        ]

        def strongest(rows):
            usable = [row for row in rows if _number(row.get("open_interest")) is not None]
            if not usable:
                return None
            return max(usable, key=lambda row: float(row.get("open_interest")))

        support_row = strongest(support_rows)
        resistance_row = strongest(resistance_rows)
        oi_candidates[str(reference)] = {
            "support": support_row.get("strike") if support_row else None,
            "resistance": resistance_row.get("strike") if resistance_row else None,
            "support_pe_oi": support_row.get("open_interest") if support_row else None,
            "resistance_ce_oi": resistance_row.get("open_interest") if resistance_row else None,
        }

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "underlying": "CRUDEOIL",
        "expiry": options[0].get("expiry"),
        "contract_count": len(result),
        "rows": result,
        "by_strike": grouped,
        "oi_candidates": oi_candidates,
        "note": "No S/R formula is applied yet; validate broker fields before bot integration.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"CRUDEOIL expiry: {payload['expiry']} | center: {center}")
    print(f"Contracts requested: {len(options)} | broker rows: {len(rows)}")
    print("STRIKE | PE OI | CE OI | PE LTP | CE LTP | OI-S | OI-R | PE depth | CE depth")
    for strike in sorted(grouped, key=float):
        pe = grouped[strike].get("PE", {})
        ce = grouped[strike].get("CE", {})
        candidate = oi_candidates[strike]
        print(
            f"{float(strike):7.2f} | {pe.get('open_interest')} | {ce.get('open_interest')} | "
            f"{pe.get('ltp')} | {ce.get('ltp')} | {candidate['support']} | {candidate['resistance']} | "
            f"{bool(pe.get('depth'))} | {bool(ce.get('depth'))}"
        )
    print(f"Saved raw broker data: {output}")


if __name__ == "__main__":
    main()