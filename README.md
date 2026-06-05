# Crypto Risk Metrics

Новый независимый проект для расчета метрик price discovery и spillover:

- `GIS` — Generalized Information Share Lien-Shrestha.
- `GFEVD` — generalized forecast error variance decomposition Pesaran-Shin с нормировкой Diebold-Yilmaz.
- `Hasbrouck proxy` — интервальная оценка Hasbrouck information share по Cholesky-порядкам.

Проект собран с нуля в текущей папке и не импортирует старый код из `D:\PyCharmProjects\Articles\btc_snp500`.

Полный развернутый математический вывод находится в [docs/math_metrics.md](docs/math_metrics.md). Проверочное соответствие формул, размерностей и кода вынесено в [docs/formula_to_code.md](docs/formula_to_code.md). Ниже в `README.md` также приведена подробная математическая карта проекта, чтобы основные формулы были видны сразу.

## Структура

```text
crypto and_risk/
  docs/
    math_metrics.md
    formula_to_code.md
    data_sources.md
  examples/
    run_metrics.py
  src/
    metrics/
      __init__.py
      common.py
      gfevd.py
      gis.py
      hasbrouck_proxy.py
    calculations/
      config.py
      panels.py
      gfevd_analysis.py
      ci_analysis.py
      run_pipeline.py
      run_comparison_artifacts.py
      run_window_gifs.py
    parsers/
      __init__.py
      assets.py
      binance.py
      binance_us.py
      checkpoint_download.py
      histdata.py
      storage.py
      download_second_data.py
      download_usdt_usd.py
      repair_binance_timestamps.py
  tests/
    metrics/
      test_metrics_smoke.py
    calculations/
      test_block_spillovers.py
      test_ci_analysis.py
      test_panels.py
      test_window_gifs.py
    parsers/
      test_assets.py
      test_binance.py
      test_binance_us.py
      test_checkpoint_download.py
      test_download_second_data.py
      test_histdata.py
      test_storage.py
  artifacts/
    figures/
    gfevd/
    reports/
  pyproject.toml
```

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

## Секундные данные

Парсеры лежат в `src/parsers/`, данные сохраняются отдельно по каждому активу в `data/second/`. Основной формат хранения — Parquet. Дневные прокси и протяжки на секундную сетку не используются.

Текущая исследовательская корзина: `BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `TRX`, `LTC`, `SPX`, `NASDAQ100`, `UDXUSD`, `EURUSD`, `USDJPY`, `USDCHF`, `GOLD`, `BRENT`.

Публичная короткая загрузка Binance 1s для криптовалют:

```powershell
python -m parsers.download_second_data --groups crypto --start 2020-01-01T00:00:00Z --end 2020-01-01T00:04:59Z
```

Для больших периодов используйте месячные Binance zip-архивы сразу в Parquet:

```powershell
python -m parsers.download_second_data --groups crypto --mode monthly-zip --year 2020 --month 1
```

Если месячный архив Binance не найден, загрузчик пробует собрать этот месяц из дневных архивов и всё равно сохраняет единый месячный parquet `*_1s_YYYYMM.parquet`.

Параллельный прогон диапазона:

```powershell
python -m parsers.download_second_data --groups crypto histdata --mode monthly-zip --start 2018-06-01T00:00:00Z --end 2026-04-30T23:59:59Z --workers 12
```

То, что реально есть на HistData и входит в текущий расчёт, подключено отдельной группой `histdata`: `UDXUSD`, `GOLD`, `BRENT`, `EURUSD`, `USDJPY`, `USDCHF`, `SPX`, `NASDAQ100`. Это FX/CFD/index/commodity-прокси, не отдельные акции:

```powershell
python -m parsers.download_second_data --groups histdata --start 2020-01-02T14:30:00Z --end 2020-01-02T14:35:00Z --workers 4
```

Подробнее: [docs/data_sources.md](docs/data_sources.md).

## Расчёт и графики

Основной прогон строит панели `1s`, `1min`, `1h`, `1d`, GFEVD-матрицы, сетевые графики, pre/post сравнения и доверительные интервалы:

```powershell
python -m calculations.run_pipeline --frequencies 1s 1min 1h 1d --rebuild-panels
python -m calculations.run_comparison_artifacts --max-windows-per-period 16
python -m calculations.run_window_gifs --frequencies 1min 1h 1d --max-frames 24 --duration-ms 700 --network-top-n 50
```

PNG/GIF лежат в `artifacts/figures/gfevd/`, таблицы и матрицы — в `artifacts/gfevd/`, отчёты — в `artifacts/reports/`.

## Минимальное использование

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

# Математическое ядро проекта

## 1. Данные и логарифмы цен

Пусть есть $n$ рынков или активов. В момент времени $t$ наблюдается вектор цен:

```math
P_t =
\begin{bmatrix}
P_{1t}\\
P_{2t}\\
\vdots\\
P_{nt}
\end{bmatrix},
\qquad P_{it}>0.
```

Все модели оцениваются на логарифмах цен:

```math
p_t=\log P_t =
\begin{bmatrix}
\log P_{1t}\\
\log P_{2t}\\
\vdots\\
\log P_{nt}
\end{bmatrix}.
```

Базовая эконометрическая предпосылка для price discovery:

```math
p_t \sim I(1),
\qquad
\Delta p_t = p_t-p_{t-1} \sim I(0).
```

Если между рынками есть коинтеграция ранга $r$, то существует матрица

```math
\beta \in \mathbb{R}^{n \times r},
```

такая что:

```math
\beta' p_t \sim I(0).
```

Число общих стохастических трендов:

```math
n-r.
```

Для классической интерпретации GIS и Hasbrouck information share как долей одного общего эффективного ценового процесса требуется:

```math
r=n-1.
```

Тогда:

```math
n-r=1,
```

то есть у системы один общий стохастический тренд.

## 2. VAR и VECM

Начинаем с VAR($p$) в уровнях лог-цен:

```math
p_t
=
A_1p_{t-1}
+A_2p_{t-2}
+\dots
+A_pp_{t-p}
+u_t,
```

где:

```math
\mathbb{E}(u_t)=0,
\qquad
\mathbb{E}(u_tu_t')=\Omega.
```

Матрица $\Omega$ — ковариационная матрица приведенных инноваций. В общем случае:

```math
\Omega =
\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \dots & \sigma_{1n}\\
\sigma_{21} & \sigma_{22} & \dots & \sigma_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
\sigma_{n1} & \sigma_{n2} & \dots & \sigma_{nn}
\end{bmatrix},
```

и внедиагональные элементы обычно не равны нулю:

```math
\sigma_{ij}\ne 0,\qquad i\ne j.
```

Это означает, что инновации рынков коррелированы. Именно поэтому для price discovery нельзя просто взять диагональные дисперсии: нужно определить, как разложить общую инновацию на ортогональные компоненты.

VAR($p$) можно переписать как VECM:

```math
\Delta p_t
=
\Pi p_{t-1}
+\Gamma_1\Delta p_{t-1}
+\Gamma_2\Delta p_{t-2}
+\dots
+\Gamma_{p-1}\Delta p_{t-p+1}
+u_t.
```

Связь между VAR и VECM:

```math
\Pi = A_1+A_2+\dots+A_p-I_n,
```

```math
\Gamma_i = -\sum_{j=i+1}^{p}A_j,
\qquad i=1,\dots,p-1.
```

Если $\mathrm{rank}(\Pi)=r$, то:

```math
\Pi=\alpha\beta',
```

где:

- $\beta$ — коинтеграционные векторы;
- $\alpha$ — коэффициенты корректировки к долгосрочному равновесию.

Компонента ошибки равновесия:

```math
z_{t-1}=\beta'p_{t-1}.
```

Тогда VECM можно читать как:

```math
\Delta p_t
=
\alpha z_{t-1}
+\sum_{i=1}^{p-1}\Gamma_i\Delta p_{t-i}
+u_t.
```

Если $z_{t-1}$ отклоняется от равновесия, матрица $\alpha$ показывает, какие рынки и с какой скоростью возвращают систему к общей долгосрочной связи.

## 3. Долгосрочная матрица воздействия

Для price discovery важен не любой краткосрочный шок, а его постоянный эффект на уровень лог-цен. Для этого используется долгосрочная матрица воздействия $C$.

Определим ортогональные дополнения:

```math
\alpha_\perp'\alpha=0,
\qquad
\beta_\perp'\beta=0.
```

Размерности:

```math
\alpha_\perp,\beta_\perp \in \mathbb{R}^{n\times(n-r)}.
```

Определим:

```math
\Gamma
=
I_n-\Gamma_1-\Gamma_2-\dots-\Gamma_{p-1}.
```

По теореме представления Грейнджера долгосрочная матрица воздействия равна:

```math
C
=
\beta_\perp
\left(
\alpha_\perp'\Gamma\beta_\perp
\right)^{-1}
\alpha_\perp'.
```

Интерпретация:

```math
p_t
=
C\sum_{s=1}^{t}u_s
+\text{стационарная часть}
+\text{детерминированная часть}.
```

Поэтому:

```math
Cu_t
```

есть постоянная компонента реакции цен на приведенную инновацию $u_t$.

Если $r=n-1$, то $n-r=1$, и матрица $C$ имеет ранг 1. В классической записи Hasbrouck часто используется строковый вектор долгосрочного воздействия:

```math
\psi.
```

Тогда постоянная инновация эффективной цены:

```math
\eta_t=\psi u_t.
```

В коде используется полная матрица $C$. При одном общем стохастическом тренде это эквивалентно использованию $\psi$, потому что все ненулевые строки $C$ пропорциональны, а итоговые доли нормируются.

# GFEVD

## 4. MA-представление

Для GFEVD нужна реакция системы на шоки на прогнозном горизонте. VAR имеет MA-представление:

```math
p_t
=
\sum_{h=0}^{\infty}\Psi_hu_{t-h}.
```

Коэффициенты считаются рекурсивно:

```math
\Psi_0=I_n,
```

```math
\Psi_h
=
A_1\Psi_{h-1}
+A_2\Psi_{h-2}
+\dots
+A_p\Psi_{h-p},
\qquad h\ge1.
```

Если индекс отрицательный, считаем:

```math
\Psi_k=0,\qquad k<0.
```

На практике берется конечный горизонт $H$:

```math
h=0,1,\dots,H-1.
```

## 5. Ошибка прогноза

Ошибка прогноза на горизонте $H$:

```math
p_{t+H}-\mathbb{E}_t(p_{t+H})
=
\sum_{h=0}^{H-1}\Psi_hu_{t+H-h}.
```

Для $i$-й переменной берем единичный вектор $e_i$. Ошибка прогноза $i$-й переменной:

```math
e_i'
\left(
p_{t+H}-\mathbb{E}_t(p_{t+H})
\right)
=
\sum_{h=0}^{H-1}e_i'\Psi_hu_{t+H-h}.
```

Ее дисперсия:

```math
D_i(H)
=
\sum_{h=0}^{H-1}
e_i'\Psi_h\Omega\Psi_h'e_i.
```

Это полный знаменатель FEVD для переменной $i$.

## 6. Generalized-шок Pesaran-Shin

Ортогональная FEVD требует Cholesky-разложения $\Omega$, но Cholesky зависит от порядка переменных. Pesaran-Shin generalized FEVD вместо этого задает шок $j$-й инновации с учетом ее корреляций со всеми остальными инновациями.

Дисперсия $j$-й инновации:

```math
\sigma_{jj}=e_j'\Omega e_j.
```

Один стандартный generalized-шок в $j$-й инновации:

```math
g_j
=
\frac{\Omega e_j}{\sqrt{\sigma_{jj}}}.
```

Реакция $i$-й переменной через $h$ периодов:

```math
e_i'\Psi_hg_j
=
\frac{e_i'\Psi_h\Omega e_j}{\sqrt{\sigma_{jj}}}.
```

Квадрат реакции:

```math
\left(e_i'\Psi_hg_j\right)^2
=
\frac{
\left(e_i'\Psi_h\Omega e_j\right)^2
}{
\sigma_{jj}
}.
```

Суммарный вклад generalized-шока $j$ в прогнозную дисперсию $i$:

```math
N_{ij}(H)
=
\sigma_{jj}^{-1}
\sum_{h=0}^{H-1}
\left(
e_i'\Psi_h\Omega e_j
\right)^2.
```

Сырая generalized FEVD:

```math
\theta_{ij}^{g}(H)
=
\frac{
N_{ij}(H)
}{
D_i(H)
}
=
\frac{
\sigma_{jj}^{-1}
\sum_{h=0}^{H-1}
\left(
e_i'\Psi_h\Omega e_j
\right)^2
}{
\sum_{h=0}^{H-1}
e_i'\Psi_h\Omega\Psi_h'e_i
}.
```

Из-за коррелированности шоков строки сырой generalized FEVD обычно не суммируются в 1:

```math
\sum_{j=1}^{n}\theta_{ij}^{g}(H)\ne1.
```

Поэтому используется строковая нормировка Diebold-Yilmaz:

```math
\widetilde{\theta}_{ij}^{g}(H)
=
\frac{
\theta_{ij}^{g}(H)
}{
\sum_{k=1}^{n}\theta_{ik}^{g}(H)
}.
```

После нормировки:

```math
\sum_{j=1}^{n}\widetilde{\theta}_{ij}^{g}(H)=1.
```

Итоговая матрица:

```math
\widetilde{\Theta}^{g}(H)
=
\left[
\widetilde{\theta}_{ij}^{g}(H)
\right]_{i,j=1}^{n}.
```

Строка $i$ отвечает на вопрос: какие шоки объясняют ошибку прогноза рынка $i$. Столбец $j$ отвечает на вопрос: куда распространяется шок рынка $j$.

## 7. Connectedness на основе GFEVD

Внедиагональная сумма строки:

```math
FROM_i
=
\sum_{\substack{j=1\\j\ne i}}^{n}
\widetilde{\theta}_{ij}.
```

Это доля прогнозной дисперсии рынка $i$, полученная от остальных рынков.

Внедиагональная сумма столбца:

```math
TO_j
=
\sum_{\substack{i=1\\i\ne j}}^{n}
\widetilde{\theta}_{ij}.
```

Это доля шока рынка $j$, переданная остальным рынкам.

Чистый spillover:

```math
NET_i=TO_i-FROM_i.
```

Общий индекс connectedness:

```math
TCI
=
\frac{1}{n}
\sum_{\substack{i,j=1\\i\ne j}}^{n}
\widetilde{\theta}_{ij}.
```

# GIS

## 8. Долгосрочная инновация эффективной цены

GIS измеряет не прогнозную дисперсию, а вклад рынка в долгосрочную инновацию общего эффективного ценового тренда.

При одном общем стохастическом тренде:

```math
\eta_t=\psi u_t.
```

Дисперсия этой инновации:

```math
\mathrm{Var}(\eta_t)
=
\mathrm{Var}(\psi u_t)
=
\psi\Omega\psi'.
```

Если бы $\Omega$ была диагональной:

```math
\Omega=\mathrm{diag}(\sigma_{11},\dots,\sigma_{nn}),
```

то вклад рынка $j$ был бы:

```math
\frac{
\psi_j^2\sigma_{jj}
}{
\sum_{k=1}^{n}\psi_k^2\sigma_{kk}
}.
```

Но в реальных данных:

```math
\Omega \text{ не диагональна}.
```

Поэтому нужно разложить коррелированные инновации $u_t$ на ортогональные шоки.

## 9. Факторизация ковариации

Пусть:

```math
\Omega=FF'.
```

Тогда можно записать:

```math
u_t=Fz_t,
\qquad
\mathbb{E}(z_tz_t')=I_n.
```

Долгосрочная инновация:

```math
\eta_t
=
\psi Fz_t.
```

Обозначим:

```math
q=\psi F.
```

Тогда:

```math
\eta_t=qz_t=\sum_{j=1}^{n}q_jz_{jt}.
```

Так как $z_{jt}$ ортогональны:

```math
\mathrm{Var}(\eta_t)
=
\sum_{j=1}^{n}q_j^2.
```

Доля $j$-го ортогонального шока:

```math
IS_j(F)
=
\frac{
q_j^2
}{
\sum_{k=1}^{n}q_k^2
}
=
\frac{
\left(\psi Fe_j\right)^2
}{
\psi\Omega\psi'
}.
```

Если $F$ — Cholesky-фактор, доля зависит от порядка переменных. GIS заменяет его на факторизацию Lien-Shrestha, не привязанную к Cholesky-порядку.

## 10. Модифицированный корень Lien-Shrestha

Разложим ковариационную матрицу инноваций через стандартные отклонения и корреляции:

```math
\Omega=DRD,
```

где:

```math
D=
\mathrm{diag}
\left(
\sqrt{\sigma_{11}},
\sqrt{\sigma_{22}},
\dots,
\sqrt{\sigma_{nn}}
\right),
```

```math
R=D^{-1}\Omega D^{-1}.
```

Матрица $R$ — корреляционная матрица инноваций.

Спектральное разложение:

```math
R=G\Lambda G',
```

где:

```math
G'G=I_n,
```

а $\Lambda$ — диагональная матрица собственных значений.

Определим модифицированный фактор:

```math
F_M
=
DG\Lambda^{1/2}G'.
```

Проверим, что он является квадратным корнем $\Omega$:

```math
F_MF_M'
=
\left(DG\Lambda^{1/2}G'\right)
\left(DG\Lambda^{1/2}G'\right)'.
```

Так как $D'=D$, $G'G=I_n$, $\Lambda^{1/2}$ симметрична:

```math
F_MF_M'
=
DG\Lambda^{1/2}G'G\Lambda^{1/2}G'D
=
DG\Lambda G'D
=
DRD
=
\Omega.
```

Значит:

```math
u_t=F_Mz_t,
\qquad
\mathbb{E}(z_tz_t')=I_n.
```

## 11. Формула GIS

Долгосрочная инновация:

```math
\eta_t=\psi F_Mz_t.
```

Пусть:

```math
q^M=\psi F_M.
```

Тогда вклад рынка $j$:

```math
GIS_j
=
\frac{
\left(q_j^M\right)^2
}{
\sum_{k=1}^{n}
\left(q_k^M\right)^2
}.
```

Так как:

```math
\sum_{k=1}^{n}
\left(q_k^M\right)^2
=
q^M(q^M)'
=
\psi F_MF_M'\psi'
=
\psi\Omega\psi',
```

получаем:

```math
GIS_j
=
\frac{
\left(\psi F_Me_j\right)^2
}{
\psi\Omega\psi'
}.
```

В реализации используется матрица $C$:

```math
Q=CF_M.
```

Тогда:

```math
GIS_j
=
\frac{
\lVert Qe_j\rVert_2^2
}{
\sum_{k=1}^{n}
\lVert Qe_k\rVert_2^2
}.
```

При $r=n-1$ эта формула эквивалентна классической формуле через $\psi$, потому что $C$ имеет ранг 1.

# Hasbrouck Proxy

## 12. Hasbrouck information share при заданном порядке

Hasbrouck information share использует Cholesky-ортогонализацию. Пусть $\pi$ — порядок переменных, а $P_\pi$ — матрица перестановки.

Ковариация в переставленном порядке:

```math
\Omega_\pi=P_\pi\Omega P_\pi'.
```

Cholesky-разложение:

```math
\Omega_\pi=L_\pi L_\pi',
```

где $L_\pi$ — нижнетреугольная матрица.

Соответствующий фактор в исходном порядке:

```math
F_\pi=P_\pi'L_\pi.
```

Проверка:

```math
F_\pi F_\pi'
=
P_\pi'L_\pi L_\pi'P_\pi
=
P_\pi'\Omega_\pi P_\pi
=
\Omega.
```

Значит, при порядке $\pi$:

```math
u_t=F_\pi z_t,
\qquad
\mathbb{E}(z_tz_t')=I_n.
```

Долгосрочная инновация:

```math
\eta_t=\psi F_\pi z_t.
```

Пусть:

```math
q^\pi=\psi F_\pi.
```

Тогда Hasbrouck information share рынка $j$ при порядке $\pi$:

```math
IS_j(\pi)
=
\frac{
\left(q_j^\pi\right)^2
}{
\sum_{k=1}^{n}
\left(q_k^\pi\right)^2
}
=
\frac{
\left(\psi F_\pi e_j\right)^2
}{
\psi\Omega\psi'
}.
```

В матричной реализации:

```math
Q_\pi=CF_\pi,
```

```math
IS_j(\pi)
=
\frac{
\lVert Q_\pi e_j\rVert_2^2
}{
\sum_{k=1}^{n}
\lVert Q_\pi e_k\rVert_2^2
}.
```

## 13. Почему получается интервал

Cholesky-фактор зависит от порядка переменных:

```math
L_\pi \ne L_{\pi'}
```

для разных $\pi$ и $\pi'$. Поэтому:

```math
IS_j(\pi)\ne IS_j(\pi').
```

Hasbrouck не дает одну единственную долю без дополнительной структурной идентификации. Вместо этого строятся границы по множеству порядков $\Pi$:

```math
IS_j^{lower}
=
\min_{\pi\in\Pi}IS_j(\pi),
```

```math
IS_j^{upper}
=
\max_{\pi\in\Pi}IS_j(\pi).
```

Средняя точка интервала:

```math
IS_j^{mid}
=
\frac{
IS_j^{lower}+IS_j^{upper}
}{2}.
```

Среднее по порядкам:

```math
IS_j^{mean}
=
\frac{1}{|\Pi|}
\sum_{\pi\in\Pi}IS_j(\pi).
```

Стандартное отклонение:

```math
IS_j^{std}
=
\sqrt{
\frac{1}{|\Pi|}
\sum_{\pi\in\Pi}
\left(
IS_j(\pi)-IS_j^{mean}
\right)^2
}.
```

Если $n$ мало, можно перебрать все порядки:

```math
|\Pi|=n!.
```

Если $n$ велико, полный перебор становится дорогим, поэтому в коде `max_orderings` ограничивает число Cholesky-порядков. Тогда результат является практической аппроксимацией Hasbrouck bounds.

## 14. Попарный Hasbrouck proxy

Для каждой пары рынков $(i,j)$ оценивается отдельная двухмерная система. В ней:

```math
n=2,
\qquad
r=1.
```

Доступно только два Cholesky-порядка:

```math
(i,j),
\qquad
(j,i).
```

Для пары строятся:

```math
IS_i^{lower},
\quad
IS_i^{upper},
\quad
IS_i^{mid}.
```

В итоговую попарную матрицу записывается:

```math
H_{ij}=IS_i^{mid},
```

```math
H_{ji}=1-IS_i^{mid},
```

```math
H_{ii}=0.5.
```

Такая матрица показывает, какой актив доминирует в price discovery в каждой паре.

# Связь формул с кодом

## 15. Файлы расчета

`src/metrics/common.py`:

- подготовка лог-цен;
- выбор лагов;
- тест Йохансена;
- оценка VECM;
- расчет $\Psi_h$;
- расчет долгосрочной матрицы $C$;
- факторизации $\Omega$;
- нормировки.

`src/metrics/gfevd.py`:

```math
\theta_{ij}^{g}(H)
=
\frac{
\sigma_{jj}^{-1}
\sum_{h=0}^{H-1}
\left(
e_i'\Psi_h\Omega e_j
\right)^2
}{
\sum_{h=0}^{H-1}
e_i'\Psi_h\Omega\Psi_h'e_i
}.
```

`src/metrics/gis.py`:

```math
GIS_j
=
\frac{
\lVert CF_Me_j\rVert_2^2
}{
\sum_k\lVert CF_Me_k\rVert_2^2
}.
```

`src/metrics/hasbrouck_proxy.py`:

```math
IS_j(\pi)
=
\frac{
\lVert CP_\pi'L_\pi e_j\rVert_2^2
}{
\sum_k\lVert CP_\pi'L_\pi e_k\rVert_2^2
}.
```

## 16. Методические источники

- Hasbrouck, J. (1995). *One Security, Many Markets: Determining the Contributions to Price Discovery*. Journal of Finance, 50(4), 1175-1199. DOI: <https://doi.org/10.1111/j.1540-6261.1995.tb04054.x>
- Pesaran, M. H., Shin, Y. (1998). *Generalized impulse response analysis in linear multivariate models*. Economics Letters, 58(1), 17-29. DOI: <https://doi.org/10.1016/S0165-1765(97)00214-0>
- Lien, D., Shrestha, K. (2009). *A new information share measure*. Journal of Futures Markets, 29(4), 377-395. DOI: <https://doi.org/10.1002/fut.20356>
- Lien, D., Shrestha, K. (2014). *Price discovery in interrelated markets*. Journal of Futures Markets, 34(3), 203-219. DOI: <https://doi.org/10.1002/fut.21593>
- Diebold, F. X., Yilmaz, K. (2012). *Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers*. International Journal of Forecasting, 28(1), 57-66. DOI: <https://doi.org/10.1016/j.ijforecast.2011.02.006>
