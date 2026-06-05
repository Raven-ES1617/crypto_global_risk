# Как считаются блоки GFEVD

Этот файл описывает, как из обычной asset-level GFEVD-матрицы строятся блоковые матрицы. Код находится в `src/calculations/gfevd_analysis.py`: функции `block_spillover_table()` и `block_matrix()`.

## 1. Что подается на вход

Сначала считается обычная GFEVD-матрица по активам:

$$
\widetilde{\Theta}^{g}(H)
=
\left[
\widetilde{\theta}^{g}_{ij}(H)
\right]_{i,j=1}^{N}.
$$

Здесь:

- $H$ - горизонт прогноза;
- $N$ - число активов;
- строка $i$ - актив, который получает шок;
- столбец $j$ - актив, откуда приходит шок;
- $\widetilde{\theta}^{g}_{ij}(H)$ - доля ошибки прогноза актива $i$, объясненная шоком актива $j$.

Матрица нормирована по строкам:

$$
\sum_{j=1}^{N}\widetilde{\theta}^{g}_{ij}(H)=1.
$$

То есть каждая строка показывает, из каких источников складывается неопределенность конкретного актива.

Пример чтения:

$$
\widetilde{\theta}^{g}_{BTC,SPX}(H)=0.12
$$

означает: 12% прогнозной дисперсии BTC на горизонте $H$ объясняется generalized-шоком SPX.

## 2. Как активы раскладываются по блокам

В проекте используются такие блоки:

| Блок | Активы |
|---|---|
| `crypto` | `BTC`, `ETH`, `BNB`, `XRP`, `ADA`, `TRX`, `LTC` |
| `equity_index` | `SPX`, `NASDAQ100` |
| `fx_dollar` | `UDXUSD`, `EURUSD`, `USDJPY`, `USDCHF` |
| `commodity` | `GOLD`, `BRENT` |

Формально каждому активу $i$ сопоставляется блок:

$$
b(i) \in \{\text{crypto},\text{equity\_index},\text{fx\_dollar},\text{commodity}\}.
$$

Для блока $A$ множество его активов:

$$
\mathcal{B}_A = \{i: b(i)=A\}.
$$

Размер блока:

$$
n_A = |\mathcal{B}_A|.
$$

## 3. Что означает блоковая ячейка

Блоковая ячейка отвечает на вопрос:

> В среднем, какая доля неопределенности активов блока-получателя объясняется шоками блока-источника?

Строка блоковой матрицы - блок-получатель. Столбец - блок-источник шока.

Например, ячейка:

$$
M_{\text{crypto},\text{equity\_index}} = 0.06
$$

означает: в среднем по криптоактивам 6% прогнозной дисперсии объясняется шоками фондовых индексов.

## 4. Сумма внутри блока

Берем все asset-level ячейки, где получатель лежит в блоке $A$, а источник шока лежит в блоке $B$:

$$
S_{A,B}
=
\sum_{i \in \mathcal{B}_A}
\sum_{j \in \mathcal{B}_B}
\widetilde{\theta}^{g}_{ij}(H).
$$

Но есть важное исключение.

Если $A=B$, то есть мы считаем влияние блока на самого себя, собственные диагональные шоки актива на самого себя убираются:

$$
\widetilde{\theta}^{g}_{ii}(H) = 0
\quad
\text{для всех } i \in \mathcal{B}_A.
$$

Тогда для диагонального блока фактически считается:

$$
S_{A,A}
=
\sum_{i \in \mathcal{B}_A}
\sum_{\substack{j \in \mathcal{B}_A \\ j \ne i}}
\widetilde{\theta}^{g}_{ij}(H).
$$

Зачем так делается: диагональ asset-level GFEVD в основном показывает собственную инерцию актива. Для межрыночных перетоков она неинтересна. Нам важнее, насколько активы внутри одного блока объясняют друг друга, а не сами себя.

## 5. Два варианта нормировки

Теперь в проекте считаются два варианта блоковой метрики.

Первый вариант - `block_total`. Это старый вариант, который показывает совокупный вклад всего блока-источника $B$ в среднюю строку блока-получателя $A$:

$$
M^{total}_{A,B}
=
\frac{S_{A,B}}{n_A}.
$$

Он отвечает на вопрос:

> Какая средняя часть прогнозной дисперсии актива из блока $A$ объясняется всеми шоками блока $B$ вместе?

Этот вариант зависит от числа активов в блоке-источнике $n_B$. Если в `crypto` 7 активов, а в `equity_index` 2, то у криптоблока больше столбцов-источников. Поэтому `crypto` может выглядеть более информативным просто потому, что это более широкий блок.

Это не ошибка, если мы хотим измерить совокупный канал:

```text
весь crypto block -> equity_index
весь fx_dollar block -> crypto
```

Но это плохо, если мы хотим честно сравнить среднюю силу одного источника внутри блока.

Второй вариант - `block_adjusted`. Он дополнительно делит на число directed asset-to-asset пар между блоками:

$$
M^{adjusted}_{A,B}
=
\frac{S_{A,B}}{q_{A,B}}.
$$

Для разных блоков:

$$
q_{A,B}=n_A n_B,
\qquad A \ne B.
$$

Для диагонального блока собственные пары $i \to i$ исключаются:

$$
q_{A,A}=n_A(n_A-1).
$$

Тогда:

$$
M^{adjusted}_{A,A}
=
\frac{
\sum_{i \in \mathcal{B}_A}
\sum_{\substack{j \in \mathcal{B}_A \\ j \ne i}}
\widetilde{\theta}^{g}_{ij}(H)
}
{n_A(n_A-1)}.
$$

Этот вариант отвечает на вопрос:

> Какова средняя сила одной directed связи из блока $B$ в блок $A$?

Именно `block_adjusted` лучше использовать, если мы спрашиваем: какой блок в среднем более информативен, без преимущества от того, что в нем больше активов.

Коротко:

| Вариант | Формула | Что показывает | Зависит от размера блока-источника |
|---|---|---|---|
| `block_total` | $S_{A,B}/n_A$ | совокупный вклад всего блока $B$ | да |
| `block_adjusted` | $S_{A,B}/q_{A,B}$ | среднюю силу одной связи $B \to A$ | намного меньше |

## 6. Что сохраняется в CSV

Функция `block_spillover_table()` сохраняет длинную таблицу:

| Колонка | Значение |
|---|---|
| `receiver_block` | блок-получатель |
| `shock_block` | блок-источник шока |
| `receiver_assets` | число активов в блоке-получателе |
| `shock_assets` | число активов в блоке-источнике |
| `pair_count` | число directed пар $q_{A,B}$ |
| `share_sum` | сумма $S_{A,B}$ |
| `average_receiver_share` | `block_total`, то есть $M^{total}_{A,B}$ |
| `average_pair_share` | `block_adjusted`, то есть $M^{adjusted}_{A,B}$ |

Функция `block_matrix()` превращает эту длинную таблицу в матрицу. По умолчанию она берет `average_receiver_share`, то есть `block_total`. Если передать `value_col="average_pair_share"`, получится `block_adjusted`.

$$
M =
\left[
M_{A,B}
\right].
$$

Файлы лежат здесь:

```text
artifacts/gfevd/matrices/block_spillovers_*.csv
artifacts/gfevd/matrices/block_matrix_*.csv
artifacts/gfevd/matrices/block_matrix_total_*.csv
artifacts/gfevd/matrices/block_matrix_adjusted_*.csv
artifacts/gfevd/periods/block_spillovers_*.csv
artifacts/gfevd/periods/block_matrix_*.csv
artifacts/gfevd/periods/block_matrix_total_*.csv
artifacts/gfevd/periods/block_matrix_adjusted_*.csv
artifacts/gfevd/window_gifs/block_flow_windows_*.csv
```

## 7. Как читать направление

Везде используется одна логика:

```text
строка = кто получает шок
столбец = откуда приходит шок
```

Например:

```text
receiver_block = equity_index
shock_block    = crypto
average_receiver_share = 0.21
```

Это читается так: в среднем 21% прогнозной дисперсии фондового блока объясняется шоками криптоблока.

Это не означает, что акции "растут из-за крипты". Это означает, что в VAR/GFEVD шоки криптоблока помогают объяснить ошибку прогноза фондового блока.

## 8. Pre/post сравнение

Для доковидного и постковидного периода считаются две блоковые матрицы для каждого варианта:

$$
M^{total,pre}_{A,B},
\qquad
M^{total,post}_{A,B},
$$

и:

$$
M^{adjusted,pre}_{A,B},
\qquad
M^{adjusted,post}_{A,B}.
$$

Разница для total-варианта:

$$
\Delta M^{total}_{A,B}
=
M^{total,post}_{A,B}-M^{total,pre}_{A,B}.
$$

Разница для adjusted-варианта:

$$
\Delta M^{adjusted}_{A,B}
=
M^{adjusted,post}_{A,B}-M^{adjusted,pre}_{A,B}.
$$

Если:

$$
\Delta M_{crypto,equity\_index} > 0,
$$

то после COVID шоки фондового блока стали сильнее объяснять криптоблок.

Если:

$$
\Delta M_{crypto,equity\_index} < 0,
$$

то после COVID эта связь ослабла.

Картинки лежат здесь:

```text
artifacts/figures/gfevd/pre_post/block_matrix_pre_post_*.png
artifacts/figures/gfevd/pre_post/block_matrix_diff_covid_minus_pre_*.png
artifacts/figures/gfevd/pre_post/block_matrix_adjusted_pre_post_*.png
artifacts/figures/gfevd/pre_post/block_matrix_adjusted_diff_covid_minus_pre_*.png
```

## 9. Доверительные интервалы по окнам

Доверительные интервалы здесь эмпирические. Это не аналитическая формула для стандартной ошибки GFEVD.

Делается так:

1. Период режется на rolling/subsample окна.
2. В каждом окне считается GFEVD.
3. В каждом окне считается длинная блоковая таблица.
4. Из нее берутся оба значения:

$$
M^{total,(k)}_{A,B},
\qquad
M^{adjusted,(k)}_{A,B}.
$$

5. Для каждой ячейки блока собирается набор значений. Для total-варианта:

$$
\{M_{A,B}^{total,(1)}, M_{A,B}^{total,(2)}, \dots, M_{A,B}^{total,(K)}\}.
$$

И отдельно для adjusted-варианта:

$$
\{M_{A,B}^{adjusted,(1)}, M_{A,B}^{adjusted,(2)}, \dots, M_{A,B}^{adjusted,(K)}\}.
$$

6. По этим значениям считаются квантили:

$$
CI_{95\%}
=
\left[
Q_{0.025}(M_{A,B}^{variant,(k)}),
Q_{0.975}(M_{A,B}^{variant,(k)})
\right].
$$

В таблицах это поля:

```text
q025
q975
```

Файлы:

```text
artifacts/gfevd/confidence/block_ci_*_*.csv
artifacts/gfevd/confidence/block_adjusted_ci_*_*.csv
```

На block pre/post картинках ячейка подписывается так:

```text
значение
[нижняя граница, верхняя граница]
```

Например:

```text
0.21
[0.00, 0.16]
```

Это значит: точечная оценка на полном периоде равна 0.21, а по rolling-окнам 95% эмпирический интервал примерно от 0.00 до 0.16.

## 10. Звездочки и p-value на diff матрицах

Для разницы post-minus-pre используется приближенная проверка по оконным оценкам.
Она считается отдельно для `block_total` и отдельно для `block_adjusted`.

Пусть для одной блоковой ячейки и одного варианта нормировки есть:

$$
\bar{M}^{pre}_{A,B}, \quad s^{pre}_{A,B}, \quad K_{pre},
$$

и:

$$
\bar{M}^{post}_{A,B}, \quad s^{post}_{A,B}, \quad K_{post}.
$$

Здесь:

- $\bar{M}$ - среднее по окнам;
- $s$ - стандартное отклонение по окнам;
- $K$ - число окон.

Разница средних:

$$
d_{A,B}
=
\bar{M}^{post}_{A,B}
-
\bar{M}^{pre}_{A,B}.
$$

Стандартная ошибка разницы:

$$
SE(d_{A,B})
=
\sqrt{
\frac{(s^{pre}_{A,B})^2}{K_{pre}}
+
\frac{(s^{post}_{A,B})^2}{K_{post}}
}.
$$

Тестовая статистика:

$$
z_{A,B}
=
\frac{d_{A,B}}{SE(d_{A,B})}.
$$

p-value считается как двухсторонняя нормальная аппроксимация:

$$
p_{A,B}
=
2\left(1-\Phi(|z_{A,B}|)\right).
$$

Звездочки:

| Звездочки | Условие |
|---|---|
| `***` | $p<0.001$ |
| `**` | $p<0.01$ |
| `*` | $p<0.05$ |
| пусто | $p \ge 0.05$ |

Важно: это не строгий структурный тест причинности. Это практичная проверка, насколько post/pre разница устойчива по rolling-окнам.

Файлы с p-value:

```text
artifacts/gfevd/confidence/block_diff_ci_covid_minus_pre_*.csv
artifacts/gfevd/confidence/block_adjusted_diff_ci_covid_minus_pre_*.csv
```

## 11. Динамика перетоков между блоками

Для GIF-окон дополнительно сохраняется динамика блоковых потоков:

```text
artifacts/gfevd/window_gifs/block_flow_windows_*.csv
artifacts/figures/gfevd/dynamics/block_flows_*.png
artifacts/figures/gfevd/dynamics/block_flows_adjusted_*.png
```

Каждая строка в `block_flow_windows_*.csv` - это одна блоковая ячейка в одном окне:

$$
M_{A,B}^{total,(k)}
\quad
\text{и}
\quad
M_{A,B}^{adjusted,(k)}.
$$

На графике `block_flows_*.png` для каждого блока-получателя рисуются total-линии:

```text
from crypto
from equity_index
from fx_dollar
from commodity
```

На графике `block_flows_adjusted_*.png` рисуются такие же линии, но уже в нормировке на одну пару активов.

То есть можно смотреть две вещи:

- `block_flows_*.png` - какой блок целиком сильнее объясняет данный блок во времени;
- `block_flows_adjusted_*.png` - где сильнее средняя связь одной пары активов.

## 12. Мини-пример

Допустим, есть три актива:

```text
BTC, ETH -> crypto
SPX      -> equity_index
```

GFEVD:

$$
\widetilde{\Theta}^{g}
=
\begin{bmatrix}
0.70 & 0.10 & 0.20\\
0.20 & 0.60 & 0.20\\
0.30 & 0.10 & 0.60
\end{bmatrix}.
$$

Строки и столбцы идут как `BTC`, `ETH`, `SPX`.

Блок `crypto <- crypto`:

$$
S_{crypto,crypto}
=
\widetilde{\theta}_{BTC,ETH}
+
\widetilde{\theta}_{ETH,BTC}
=
0.10 + 0.20 = 0.30.
$$

Total-вариант: делим на число крипто-получателей:

$$
M^{total}_{crypto,crypto}
=
\frac{0.30}{2}
=
0.15.
$$

Adjusted-вариант: делим на число directed пар без собственных пар:

$$
q_{crypto,crypto}=2(2-1)=2.
$$

$$
M^{adjusted}_{crypto,crypto}
=
\frac{0.30}{2}
=
0.15.
$$

Блок `crypto <- equity_index`:

$$
S_{crypto,equity\_index}
=
\widetilde{\theta}_{BTC,SPX}
+
\widetilde{\theta}_{ETH,SPX}
=
0.20 + 0.20 = 0.40.
$$

Total-вариант:

$$
M^{total}_{crypto,equity\_index}
=
\frac{0.40}{2}
=
0.20.
$$

Adjusted-вариант:

$$
q_{crypto,equity\_index}=2 \cdot 1 = 2.
$$

$$
M^{adjusted}_{crypto,equity\_index}
=
\frac{0.40}{2}
=
0.20.
$$

В этом игрушечном примере у `equity_index` только один актив, поэтому total и adjusted совпали. В нашем полном наборе они будут отличаться сильнее: у `crypto` 7 активов, у `fx_dollar` 4, у `equity_index` и `commodity` по 2.

Главное чтение:

- `block_total`: в среднем 20% прогнозной дисперсии криптоактивов объясняется всем фондовым блоком;
- `block_adjusted`: средняя directed связь `equity_index -> crypto` равна 20% на одну пару активов.
