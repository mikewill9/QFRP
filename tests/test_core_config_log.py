"""Settings resolution layers and logging idempotence."""

from __future__ import annotations

import logging
from pathlib import Path

from quantlab.config import Settings, get_settings, reset_settings
from quantlab.log import configure_logging, get_logger


def test_defaults_and_with_root(tmp_path):
    s = Settings().with_root(tmp_path).ensure_dirs()
    assert s.data_dir == tmp_path / "data" and s.data_dir.is_dir()
    assert s.cache_dir.is_dir() and s.results_dir.is_dir()
    assert s.log_level == "INFO"


def test_env_overrides_file(tmp_path, monkeypatch):
    cfg = tmp_path / "quantlab.toml"
    cfg.write_text(
        '[quantlab]\ncache_dir = "/from/file"\nlog_level = "DEBUG"\nrandom_seed = 99\n'
    )
    monkeypatch.setenv("QUANTLAB_CONFIG", str(cfg))
    monkeypatch.setenv("QUANTLAB_CACHE_DIR", str(tmp_path / "from_env"))
    reset_settings()
    try:
        s = get_settings()
        assert s.cache_dir == tmp_path / "from_env"  # env beats file
        assert s.log_level == "DEBUG"  # file beats default
        assert s.random_seed == 99
        assert isinstance(s.cache_dir, Path)
    finally:
        reset_settings()


def test_settings_cached_until_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTLAB_LOG_LEVEL", "WARNING")
    reset_settings()
    try:
        assert get_settings().log_level == "WARNING"
        monkeypatch.setenv("QUANTLAB_LOG_LEVEL", "ERROR")
        assert get_settings().log_level == "WARNING"  # still cached
        reset_settings()
        assert get_settings().log_level == "ERROR"
    finally:
        reset_settings()


def test_configure_logging_idempotent():
    logger = configure_logging("INFO")
    n = len(logger.handlers)
    configure_logging("INFO")
    assert len(logger.handlers) == n  # no handler duplication


def test_get_logger_namespacing():
    assert get_logger("quantlab.data.cache").name == "quantlab.data.cache"
    assert get_logger("myscript").name == "quantlab.myscript"
    assert isinstance(get_logger("x"), logging.Logger)
