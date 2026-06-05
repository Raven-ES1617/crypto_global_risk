from __future__ import annotations

from parsers.assets import (
    DEFAULT_CRYPTO_ASSETS,
    DEFAULT_EQUITY_ASSETS,
    DEFAULT_HISTDATA_ASSETS,
    DEFAULT_MACRO_ASSETS,
    DEFAULT_STABLECOIN_ASSETS,
    get_default_assets,
)


def test_crypto_assets_match_research_basket() -> None:
    symbols = {asset.symbol for asset in DEFAULT_CRYPTO_ASSETS}

    assert symbols == {"BTC", "ETH", "BNB", "XRP", "ADA", "TRX", "LTC"}
    assert "DOGE" not in symbols
    assert "LINK" not in symbols


def test_default_assets_exclude_manual_equity_watchlist() -> None:
    assert DEFAULT_EQUITY_ASSETS == ()
    assert get_default_assets(["equity"]) == []


def test_macro_assets_use_histdata_proxies() -> None:
    symbols = {asset.symbol for asset in DEFAULT_MACRO_ASSETS}
    assert symbols == {"UDXUSD", "GOLD", "BRENT"}


def test_stablecoin_group_keeps_only_usdt_usd() -> None:
    assert {asset.symbol for asset in DEFAULT_STABLECOIN_ASSETS} == {"USDT"}
    assert {asset.provider for asset in DEFAULT_STABLECOIN_ASSETS} == {"binance_us"}


def test_histdata_assets_are_only_available_proxies() -> None:
    assets = get_default_assets(["histdata"])
    symbols = {asset.symbol for asset in assets}

    assert assets == list(DEFAULT_HISTDATA_ASSETS)
    assert {"UDXUSD", "GOLD", "BRENT", "EURUSD", "USDJPY", "USDCHF", "SPX", "NASDAQ100"}.issubset(symbols)
    assert "OIL" not in symbols
    assert "VIX" not in symbols
    assert "UST10Y" not in symbols


def test_histdata_fx_group_contains_selected_global_fx() -> None:
    assets = get_default_assets(["histdata_fx"])
    assert {asset.symbol for asset in assets} == {"EURUSD", "USDJPY", "USDCHF"}
    assert {asset.provider for asset in assets} == {"histdata"}
