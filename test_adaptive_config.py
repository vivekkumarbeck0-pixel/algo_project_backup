import json
from types import SimpleNamespace

import pytest

from engine.adaptive_config import (
    AdaptiveConfigError,
    AdaptiveConfigReloader,
    apply_parameters,
    publish_payload,
)


def _settings():
    return SimpleNamespace(
        crude_atr_multiplier=2.0,
        buffer_points=4.0,
        crude_risk_reward_ratio=2.0,
        crude_trailing_activation_points=8.0,
        crude_min_entry_volume=0.0,
        crude_min_abs_oi_change=0.0,
        crude_min_entry_atr=0.0,
        crude_max_entry_atr=1_000_000.0,
        crude_allowed_regimes=("TRENDING", "SIDEWAYS", "UNKNOWN"),
    )


def test_reloader_applies_only_valid_config_while_flat(tmp_path):
    path = tmp_path / "adaptive.json"
    settings = _settings()
    publish_payload(path, {"crude_atr_multiplier": 1.5}, {"win_rate": 0.61})
    reloader = AdaptiveConfigReloader(path)

    assert not reloader.refresh(settings, position_is_open=True)
    assert settings.crude_atr_multiplier == 2.0
    assert reloader.refresh(settings, position_is_open=False)
    assert settings.crude_atr_multiplier == 1.5
    assert not reloader.refresh(settings, position_is_open=False)


def test_reloader_rejects_out_of_range_config_without_mutation(tmp_path):
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "status": "approved",
        "parameters": {"crude_atr_multiplier": 99},
    }), encoding="utf-8")
    settings = _settings()

    with pytest.raises(AdaptiveConfigError):
        AdaptiveConfigReloader(path).refresh(settings, position_is_open=False)

    assert settings.crude_atr_multiplier == 2.0


def test_apply_parameters_is_transactional_when_setting_is_missing():
    settings = _settings()

    with pytest.raises(AdaptiveConfigError):
        apply_parameters(settings, {
            "crude_atr_multiplier": 1.5,
            "missing_setting": 10.0,
        })

    assert settings.crude_atr_multiplier == 2.0