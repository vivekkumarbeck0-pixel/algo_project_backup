import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
import requests

from angel_one.instrument_reader import InstrumentReader


@pytest.fixture
def master(tmp_path, monkeypatch):
    path = tmp_path / "master.json"
    records = [
        {"name": "CRUDEOIL", "exch_seg": "MCX", "instrumenttype": "FUTCOM",
         "symbol": "CRUDEOILFUT", "expiry": "31DEC2099", "token": "1"},
        {"name": "CRUDEOIL", "exch_seg": "MCX", "instrumenttype": "OPTFUT",
         "symbol": "CRUDEOIL1000CE", "expiry": "31DEC2099", "strike": "100000", "token": "2"},
        {"name": "CRUDEOIL", "exch_seg": "MCX", "instrumenttype": "OPTFUT",
         "symbol": "CRUDEOIL1000PE", "expiry": "31DEC2099", "strike": "100000", "token": "3"},
        {"name": "CRUDEOILM", "exch_seg": "MCX", "instrumenttype": "FUTCOM", "token": "4"},
        {"name": "NIFTY", "exch_seg": "NFO", "instrumenttype": "OPTIDX",
         "symbol": "NIFTY1000CE", "expiry": "31DEC2099", "token": "5"},
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(InstrumentReader, "MASTER_FILE", path)
    monkeypatch.setattr(InstrumentReader, "_crude_cache_key", None)
    monkeypatch.setattr(InstrumentReader, "_crude_cache", None)
    monkeypatch.setattr(InstrumentReader, "_crude_refresh_retry_at", 0.0)
    return path, records


def test_crude_stream_filters_contracts_and_shares_cache(master, monkeypatch):
    _, records = master
    reads = []
    original = InstrumentReader._read_crude_instruments

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(InstrumentReader, "_read_crude_instruments", staticmethod(read))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: InstrumentReader(crude_only=True).load(), range(8)))

    assert len(reads) == 1
    assert all(result is results[0] for result in results)
    assert results[0] == records[:3]
    reader = InstrumentReader(crude_only=True)
    assert reader.find_nearest_future("CRUDEOIL", "FUTCOM", "MCX")["token"] == "1"
    assert reader.find_option_token("CRUDEOIL", 1000, "CE", instrument_type="OPTFUT", exchange="MCX")["token"] == "2"
    assert reader.find_option_token("CRUDEOIL", 1000, "PE", instrument_type="OPTFUT", exchange="MCX")["token"] == "3"
    assert InstrumentReader().load() == records


def test_crude_cache_reloads_when_master_changes(master):
    path, records = master
    reader = InstrumentReader(crude_only=True)
    assert reader.find_nearest_future("CRUDEOIL", "FUTCOM", "MCX")["token"] == "1"
    records[0]["token"] = "new-token"
    path.write_text(json.dumps(records), encoding="utf-8")
    assert reader.find_nearest_future("CRUDEOIL", "FUTCOM", "MCX")["token"] == "new-token"


def test_market_data_switches_between_crude_and_nifty_readers(master):
    from angel_one.market_data import MarketDataFetcher

    fetcher = MarketDataFetcher(client=None, use_env=False)
    assert fetcher._resolve_listed_option_expiry("CRUDEOIL", "2099-12-31") == "31DEC2099"
    assert fetcher._instrument_reader.crude_only
    assert fetcher._resolve_listed_option_expiry("NIFTY", "2099-12-31") == "31DEC2099"
    assert not fetcher._instrument_reader.crude_only
    assert fetcher._resolve_listed_option_expiry("CRUDEOIL", "2099-12-31") == "31DEC2099"
    assert fetcher._instrument_reader.crude_only


def test_stale_master_refresh_streams_and_reuses_result(master, monkeypatch):
    path, records = master
    yesterday = time.time() - 172800
    os.utime(path, (yesterday, yesterday))
    records[0]["token"] = "refreshed"
    payload = json.dumps(records).encode()
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield payload[:40]
            yield payload[40:]

    def get(url, timeout, stream):
        calls.append((timeout, stream))
        return Response()

    monkeypatch.setattr(requests, "get", get)
    reader = InstrumentReader(crude_only=True)
    assert reader.find_nearest_future("CRUDEOIL", "FUTCOM", "MCX")["token"] == "refreshed"
    assert len(InstrumentReader(crude_only=True).load()) == 3
    assert calls == [(30, True)]
    assert json.loads(path.read_text(encoding="utf-8")) == records
    assert datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date()


def test_failed_refresh_uses_existing_master_and_retries(master, monkeypatch, capsys):
    path, _ = master
    yesterday = time.time() - 172800
    os.utime(path, (yesterday, yesterday))
    calls = []

    def fail(url, timeout, stream):
        calls.append(1)
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fail)
    reader = InstrumentReader(crude_only=True)
    assert reader.find_nearest_future("CRUDEOIL", "FUTCOM", "MCX")["token"] == "1"
    assert len(reader.load()) == 3
    assert len(calls) == 1
    assert "MASTER REFRESH FAILED" in capsys.readouterr().out
    InstrumentReader._crude_refresh_retry_at = 0.0
    assert len(reader.load()) == 3
    assert len(calls) == 2


def test_invalid_download_does_not_replace_valid_stale_master(master, monkeypatch):
    path, records = master
    yesterday = time.time() - 172800
    os.utime(path, (yesterday, yesterday))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'{"error": "invalid master"}'

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    assert len(InstrumentReader(crude_only=True).load()) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == records
    assert not list(path.parent.glob("*.tmp"))


def test_missing_master_download_failure_is_reported(master, monkeypatch):
    path, _ = master
    path.unlink()

    def fail(url, timeout, stream):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(requests.ConnectionError, match="offline"):
        InstrumentReader(crude_only=True).load()
