# Гипотезы и какие артефакты их проверяют

Этот файл нужен как карта проекта: какая гипотеза чем проверяется и где лежат графики.

## 1. После COVID интеграция крипты с глобальными рынками стала глубже

Что смотрим:

- `TCI` - общий уровень связности системы;
- GFEVD-матрицы - кто чьи шоки объясняет;
- block GFEVD - перетоки между группами активов.

Главные файлы:

```text
artifacts/figures/gfevd/rolling_tci_*.png
artifacts/figures/gfevd/pre_post/matrix_pre_post_*.png
artifacts/figures/gfevd/pre_post/block_matrix_pre_post_*.png
artifacts/figures/gfevd/pre_post/block_matrix_adjusted_pre_post_*.png
artifacts/figures/gfevd/pre_post/block_matrix_diff_covid_minus_pre_*.png
artifacts/figures/gfevd/pre_post/block_matrix_adjusted_diff_covid_minus_pre_*.png
artifacts/stability/tci_stability.csv
artifacts/stability/tci_diff_stability.csv
artifacts/stability/block_diff_stability.csv
```

Как читать:

- если post выше pre, связь усилилась;
- если на diff-картинке есть звезды, изменение устойчиво по окнам;
- `block_matrix_adjusted_*` важен, чтобы размер криптоблока не давал ему автоматическое преимущество.

## 2. Интеграция режимная, а не постоянная

Что смотрим:

- rolling TCI;
- GIF-матрицы и GIF-сети;
- динамику block flows.

Главные файлы:

```text
artifacts/figures/gfevd/rolling_tci_*.png
artifacts/figures/gfevd/window_gifs/matrix_windows_*.gif
artifacts/figures/gfevd/window_gifs/matrix_windows_log_*.gif
artifacts/figures/gfevd/networks/window_gifs/network_windows_*.gif
artifacts/figures/gfevd/dynamics/block_flows_*.png
artifacts/figures/gfevd/dynamics/block_flows_adjusted_*.png
artifacts/figures/stability/window_block_net_flows_*.png
artifacts/figures/stability/window_crypto_global_gfevd_*.png
artifacts/stability/window_block_net_flows.csv
artifacts/stability/window_crypto_global_gfevd_flows.csv
```

Как читать:

- ровная линия означает стабильную интеграцию;
- волны и смена лидирующих связей означают режимность;
- adjusted dynamics показывает среднюю силу одной связи, а не общий размер блока.

## 3. Одни и те же инвесторы или разные сегменты рынка

Идея:

- если рынки стали ближе, должны появляться устойчивые двусторонние перетоки `crypto <-> global`;
- если крипта остается отдельным сегментом, основная масса связей останется внутри `crypto`.

Что смотрим:

- block flows из GFEVD;
- сети GFEVD;
- long-run price discovery через GIS и Hasbrouck proxy.

Главные файлы:

```text
artifacts/figures/gfevd/dynamics/block_flows_*.png
artifacts/figures/gfevd/dynamics/block_flows_adjusted_*.png
artifacts/figures/gfevd/networks/network_pre_covid_*.png
artifacts/figures/gfevd/networks/network_covid_and_after_*.png
artifacts/figures/price_discovery/long_run/crypto_global_pre_post_*.png
artifacts/figures/price_discovery/long_run/crypto_global_diff_covid_minus_pre_*.png
artifacts/figures/price_discovery/long_run/crypto_global_dynamics_*.png
artifacts/figures/stability/crypto_global_gfevd_*.png
artifacts/figures/stability/block_net_flows_*.png
artifacts/stability/crypto_global_gfevd_flows.csv
artifacts/stability/block_net_flows.csv
```

Как читать:

- GFEVD говорит, кто передает прогнозные шоки на заданном горизонте;
- GIS и Hasbrouck proxy говорят, кто сильнее участвует в долгосрочном price discovery;
- если оба слоя дают похожую картину, аргумент сильнее.

## 4. Кто ведет долгосрочную цену: крипта или глобальные рынки

Здесь нужны GIS и Hasbrouck proxy.

Обе метрики отвечают на другой вопрос, чем GFEVD:

```text
GFEVD: чьи шоки объясняют ошибку прогноза на горизонте H?
GIS/Hasbrouck: кто вносит долгосрочную информацию в общий ценовой тренд?
```

В наших pairwise-таблицах левая сторона пары - криптоактив:

```text
BTC-SPX
BTC-NASDAQ100
ETH-SPX
ETH-NASDAQ100
BTC-UDXUSD
ETH-UDXUSD
BTC-GOLD
BTC-BRENT
```

Поэтому `left_share` читается как крипто-доля в долгосрочном price discovery пары.

Главные файлы:

```text
artifacts/figures/price_discovery/long_run/pair_pre_post_*.png
artifacts/figures/price_discovery/long_run/pair_diff_covid_minus_pre_*.png
artifacts/price_discovery/summary/long_run_pair_ci_*.csv
artifacts/price_discovery/summary/long_run_pair_diff_covid_minus_pre_*.csv
artifacts/stability/price_discovery_pair_stability.csv
artifacts/stability/price_discovery_pair_diff_stability.csv
artifacts/stability/price_discovery_metric_agreement.csv
```

Как читать:

- значение около `0.5` - паритет;
- выше `0.5` - криптосторона пары сильнее ведет долгосрочную цену;
- ниже `0.5` - глобальный актив сильнее ведет долгосрочную цену;
- post-minus-pre выше нуля означает, что после COVID крипта стала сильнее как long-run price discovery source.

## 5. Частота имеет смысл

Частоты:

```text
1s, 1min, 1h, 1d
```

Как читать:

- эффект на `1min` и `1h` ближе к быстрой передаче информации;
- эффект только на `1d` больше похож на общий risk-on/risk-off режим;
- если GIS/Hasbrouck устойчивы на `1h` и `1d`, это аргумент про долгосрочное лидерство, а не только шум.

Главные файлы:

```text
artifacts/figures/gfevd/blocks_full_*.png
artifacts/figures/gfevd/blocks_adjusted_full_*.png
artifacts/figures/price_discovery/long_run/pair_pre_post_*.png
artifacts/figures/price_discovery/long_run/crypto_global_dynamics_*.png
artifacts/figures/stability/frequency_effects_heatmap.png
artifacts/stability/frequency_consistency.csv
```

## 6. Общая проверка устойчивости

Чтобы не делать выводы только по одной красивой картинке, добавлен отдельный stability-слой.

Что там проверяется:

- достаточно ли rolling-окон;
- насколько широкие доверительные интервалы;
- исключает ли post-minus-pre интервал ноль;
- есть ли значимость по p-value;
- совпадает ли знак эффекта на `1min`, `1h`, `1d`;
- согласны ли GIS и Hasbrouck по сторону от паритета `0.5`.

Главные файлы:

```text
artifacts/reports/stability_summary.md
artifacts/reports/stability_manifest.json
artifacts/stability/hypothesis_stability_checks.csv
artifacts/stability/frequency_effects.csv
artifacts/stability/frequency_consistency.csv
artifacts/figures/stability/hypothesis_stability_dashboard.png
artifacts/figures/stability/frequency_effects_heatmap.png
```
