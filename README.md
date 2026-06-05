# Crypto Global Risk

Инструменты для расчета метрик price discovery и spillover на криптовалютных и глобальных финансовых рынках.

Проект считает:

- `GIS` - Generalized Information Share Lien-Shrestha;
- `GFEVD` - generalized forecast error variance decomposition Pesaran-Shin с нормировкой Diebold-Yilmaz;
- `Hasbrouck proxy` - интервальную оценку Hasbrouck information share по Cholesky-порядкам;
- block-level connectedness для групп активов.

Подробная математика и связь формул с кодом вынесены отдельно:

- [docs/math_metrics.md](docs/math_metrics.md) - вывод GIS, GFEVD и Hasbrouck proxy;
- [docs/formula_to_code.md](docs/formula_to_code.md) - соответствие формул реализациям;
- [docs/block_gfevd.md](docs/block_gfevd.md) - блоковые GFEVD-матрицы;
- [docs/data_sources.md](docs/data_sources.md) - источники и загрузка данных;
- [docs/hypotheses_and_artifacts.md](docs/hypotheses_and_artifacts.md) - исследовательские гипотезы и артефакты.

## Структура

```text
crypto_global_risk/
  docs/
  examples/
  src/
    calculations/
    metrics/
    parsers/
  tests/
  pyproject.toml
```

`data/`, `artifacts/`, текст статьи, шаблоны журналов и LaTeX/PDF-сборки не входят в репозиторий.

## Установка

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

## Быстрый запуск

```powershell
python examples\run_metrics.py
pytest
```

## Данные

Парсеры находятся в `src/parsers/`, локальные данные сохраняются в `data/second/`.
Основной формат хранения - Parquet.

Текущая исследовательская корзина:

```text
BTC, ETH, BNB, XRP, ADA, TRX, LTC,
SPX, NASDAQ100, UDXUSD, EURUSD, USDJPY, USDCHF,
GOLD, BRENT
```

Короткая загрузка Binance 1s:

```powershell
python -m parsers.download_second_data --groups crypto --start 2020-01-01T00:00:00Z --end 2020-01-01T00:04:59Z
```

Загрузка месячных архивов Binance сразу в Parquet:

```powershell
python -m parsers.download_second_data --groups crypto --mode monthly-zip --year 2020 --month 1
```

Параллельная загрузка диапазона:

```powershell
python -m parsers.download_second_data --groups crypto histdata --mode monthly-zip --start 2018-06-01T00:00:00Z --end 2026-04-30T23:59:59Z --workers 12
```

HistData-группа включает FX/CFD/index/commodity-прокси:

```text
UDXUSD, GOLD, BRENT, EURUSD, USDJPY, USDCHF, SPX, NASDAQ100
```

Подробнее: [docs/data_sources.md](docs/data_sources.md).

## Расчеты

Основной прогон строит панели `1s`, `1min`, `1h`, `1d`, GFEVD-матрицы, сетевые графики, pre/post сравнения и доверительные интервалы:

```powershell
python -m calculations.run_pipeline --frequencies 1s 1min 1h 1d --rebuild-panels
python -m calculations.run_comparison_artifacts --max-windows-per-period 16
python -m calculations.run_window_gifs --frequencies 1min 1h 1d --max-frames 24 --duration-ms 700 --network-top-n 50
```

Локальные выходы:

- `artifacts/figures/gfevd/` - PNG/GIF;
- `artifacts/gfevd/` - таблицы и матрицы;
- `artifacts/reports/` - отчеты и manifest-файлы.

## Минимальный пример

```python
import pandas as pd

from metrics import (
    calculate_gfevd,
    calculate_gis,
    calculate_hasbrouck_proxy,
)

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

gis = calculate_gis(prices)
gfevd = calculate_gfevd(prices, horizon=20)
hasbrouck = calculate_hasbrouck_proxy(prices)

print(gis.shares)
print(gfevd.table)
print(hasbrouck.summary)
```

## Методические источники

- Hasbrouck, J. (1995). *One Security, Many Markets: Determining the Contributions to Price Discovery*. Journal of Finance, 50(4), 1175-1199. DOI: <https://doi.org/10.1111/j.1540-6261.1995.tb04054.x>
- Pesaran, M. H., Shin, Y. (1998). *Generalized impulse response analysis in linear multivariate models*. Economics Letters, 58(1), 17-29. DOI: <https://doi.org/10.1016/S0165-1765(97)00214-0>
- Lien, D., Shrestha, K. (2009). *A new information share measure*. Journal of Futures Markets, 29(4), 377-395. DOI: <https://doi.org/10.1002/fut.20356>
- Lien, D., Shrestha, K. (2014). *Price discovery in interrelated markets*. Journal of Futures Markets, 34(3), 203-219. DOI: <https://doi.org/10.1002/fut.21593>
- Diebold, F. X., Yilmaz, K. (2012). *Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers*. International Journal of Forecasting, 28(1), 57-66. DOI: <https://doi.org/10.1016/j.ijforecast.2011.02.006>
