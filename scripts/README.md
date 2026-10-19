FOLDER : scripts

Purpose :

One-off/manual helper scripts used during development and API
exploration. These are not part of the main capture -> parse ->
decide pipeline and are not covered by automated tests.

Files :

demo_market_to_aoc.py

Manual demo wiring market data into the AOC engine.

exhaustive_payload_probe.py / try_payload_variants.py

Exploratory scripts used to probe Angel One SmartAPI request/response
payload shapes.

inspect_smartapi.py

Inspects the installed `SmartApi` package (methods, signatures) to
help debug integration issues.

probe_token_exchanges.py

Helper for resolving instrument tokens/exchanges via the Angel One
instrument master.

Status :

These are developer utilities, not production code. Prefer promoting
any logic that becomes load-bearing into `angel_one/` or `engine/`
with proper logging and error handling.
