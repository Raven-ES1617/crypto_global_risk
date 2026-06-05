from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    provider: str
    source_symbol: str
    group: str
    kind: str
    sector: str | None = None
    description: str | None = None


DEFAULT_CRYPTO_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("BTC", "binance", "BTCUSDT", "crypto", "spot_crypto", description="Bitcoin / USDT"),
    AssetSpec("ETH", "binance", "ETHUSDT", "crypto", "spot_crypto", description="Ethereum / USDT"),
    AssetSpec("BNB", "binance", "BNBUSDT", "crypto", "spot_crypto", description="BNB / USDT"),
    AssetSpec("XRP", "binance", "XRPUSDT", "crypto", "spot_crypto", description="XRP / USDT"),
    AssetSpec("ADA", "binance", "ADAUSDT", "crypto", "spot_crypto", description="Cardano / USDT"),
    AssetSpec("TRX", "binance", "TRXUSDT", "crypto", "spot_crypto", description="TRON / USDT"),
    AssetSpec("LTC", "binance", "LTCUSDT", "crypto", "spot_crypto", description="Litecoin / USDT"),
)

DEFAULT_STABLECOIN_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("USDT", "binance_us", "USDTUSD", "stablecoin", "spot_stablecoin", description="Tether / USD"),
)

DEFAULT_EQUITY_ASSETS: tuple[AssetSpec, ...] = ()

DEFAULT_HISTDATA_MACRO_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("UDXUSD", "histdata", "udxusd", "histdata_macro", "index_cfd", "Dollar", "US Dollar Index CFD proxy"),
    AssetSpec("GOLD", "histdata", "xauusd", "histdata_macro", "commodity_cfd", "Metals", "Gold CFD proxy"),
    AssetSpec("BRENT", "histdata", "bcousd", "histdata_macro", "commodity_cfd", "Energy", "Brent oil CFD proxy"),
)

DEFAULT_HISTDATA_FX_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("EURUSD", "histdata", "eurusd", "histdata_fx", "fx_spot", "FX", "Euro / US dollar"),
    AssetSpec("USDJPY", "histdata", "usdjpy", "histdata_fx", "fx_spot", "FX", "US dollar / Japanese yen"),
    AssetSpec("USDCHF", "histdata", "usdchf", "histdata_fx", "fx_spot", "FX", "US dollar / Swiss franc"),
)

DEFAULT_HISTDATA_INDEX_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("SPX", "histdata", "spxusd", "histdata_index", "index_cfd", "Equity Index", "S&P 500 CFD proxy"),
    AssetSpec(
        "NASDAQ100",
        "histdata",
        "nsxusd",
        "histdata_index",
        "index_cfd",
        "Equity Index",
        "Nasdaq 100 CFD proxy",
    ),
)

DEFAULT_HISTDATA_ASSETS: tuple[AssetSpec, ...] = (
    DEFAULT_HISTDATA_MACRO_ASSETS + DEFAULT_HISTDATA_FX_ASSETS + DEFAULT_HISTDATA_INDEX_ASSETS
)
DEFAULT_MACRO_ASSETS: tuple[AssetSpec, ...] = DEFAULT_HISTDATA_MACRO_ASSETS


def get_default_assets(groups: tuple[str, ...] | list[str] | None = None) -> list[AssetSpec]:
    selected = set(groups or ("crypto", "histdata"))
    assets: list[AssetSpec] = []
    if "crypto" in selected:
        assets.extend(DEFAULT_CRYPTO_ASSETS)
    if "stablecoin" in selected or "stablecoins" in selected:
        assets.extend(DEFAULT_STABLECOIN_ASSETS)
    if "equity" in selected or "equities" in selected:
        assets.extend(DEFAULT_EQUITY_ASSETS)
    if "macro" in selected:
        assets.extend(DEFAULT_MACRO_ASSETS)
    if "histdata" in selected:
        assets.extend(DEFAULT_HISTDATA_ASSETS)
    if "histdata_macro" in selected or "macro_histdata" in selected:
        assets.extend(DEFAULT_HISTDATA_MACRO_ASSETS)
    if "histdata_fx" in selected or "fx" in selected:
        assets.extend(DEFAULT_HISTDATA_FX_ASSETS)
    if "histdata_index" in selected or "index_proxy" in selected:
        assets.extend(DEFAULT_HISTDATA_INDEX_ASSETS)
    return assets
