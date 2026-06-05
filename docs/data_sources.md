# Second-Level Data Sources

Проект хранит только настоящие секундные ряды. Дневные прокси и протяжки на секундную сетку здесь не используются.

Файлы лежат отдельно по каждому активу:

```text
data/
  second/
    binance/
      BTC/
        BTC_1s_202001.parquet
    histdata/
      SPX/
        SPX_1s_202001.parquet
```

`data/` находится в `.gitignore`, поэтому большие файлы не попадают в репозиторий.

## Активы

Crypto из Binance:

- `BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `TRX`, `LTC`.

USDT/USD держится отдельно, только если он явно нужен для проверки Tether:

- `USDTUSD` через отдельный скрипт `python -m parsers.download_usdt_usd`.

HistData-прокси:

- macro/commodities: `UDXUSD`, `GOLD` (`XAUUSD`), `BRENT` (`BCOUSD`);
- FX: `EURUSD`, `USDJPY`, `USDCHF`;
- index proxies: `SPX` (`SPXUSD`), `NASDAQ100` (`NSXUSD`).

## Binance

Для коротких проверок можно использовать public REST API `GET /api/v3/klines` с interval `1s`:

```powershell
python -m parsers.download_second_data --groups crypto --start 2020-01-01T00:00:00Z --end 2020-01-01T00:04:59Z
```

Для больших периодов используется `data.binance.vision`: сначала месячный архив, потом fallback на дневные `1s` архивы. Результат всегда пишется как месячный parquet:

```powershell
python -m parsers.download_second_data --groups crypto histdata --mode monthly-zip --start 2018-06-01T00:00:00Z --end 2026-04-30T23:59:59Z --workers 12
```

## HistData

HistData используется только для доступных tick-last CFD/index/commodity-прокси. Архив скачивается по месяцу, тики агрегируются в секундные OHLCV-бары и сохраняются в Parquet.

```powershell
python -m parsers.download_second_data --groups histdata --start 2020-01-02T14:30:00Z --end 2020-01-02T14:35:00Z --workers 4
```

Время HistData трактуется как EST без daylight saving adjustment и переводится в UTC.
