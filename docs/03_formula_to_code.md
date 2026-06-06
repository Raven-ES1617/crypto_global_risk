# Формулы и соответствие коду

Этот файл показывает одну простую вещь: какая формула за что отвечает в коде.

Я оставляю формулы, но пояснения пишу проще. Смысл такой:

1. Сначала мы готовим логарифмы цен.
2. Потом оцениваем VECM.
3. Из VECM берем матрицы, которые нужны для GIS, GFEVD и Hasbrouck proxy.
4. Потом считаем сами метрики.

## 1. Что подается на вход

Пусть есть:

- $T$ — число наблюдений.
- $n$ — число активов.
- $p$ — число лагов VAR в уровнях.

В код приходит таблица цен:

```math
P \in \mathbb{R}^{T \times n}
```

Здесь строки — даты, столбцы — активы.

Дальше цены переводятся в логарифмы:

```math
p_t = \log P_t
```

Для всей таблицы:

```math
X \in \mathbb{R}^{T \times n}
```

Здесь $X$ — таблица лог-цен.

В коде это делается так:

```python
log_prices = np.log(price_data)
```

Если уже переданы лог-цены, используется параметр:

```python
input_is_log=True
```

Тогда повторный логарифм не берется.

## 2. Первые разности

Для выбора лагов используются не сами лог-цены, а их изменения:

```math
\Delta p_t = p_t - p_{t-1}
```

Таблица первых разностей имеет размер:

```math
\Delta X \in \mathbb{R}^{(T-1) \times n}
```

В коде:

```python
diff = log_prices.diff().dropna()
VAR(diff).select_order(maxlags=max_lags)
```

Результат выбора лагов обозначим так:

```math
k = k_{\mathrm{diff}}
```

Здесь $k$ — число лагов разностей в VECM.

Если VECM использует $k$ лагов разностей, то соответствующая VAR-модель в уровнях имеет порядок:

```math
p = k + 1
```

## 3. VECM

Основная модель:

```math
\Delta p_t
=
\alpha \beta' p_{t-1}
+
\sum_{\ell=1}^{k}
\Gamma_\ell \Delta p_{t-\ell}
+
u_t
```

Что здесь означает каждая часть:

- $\Delta p_t$ — изменение лог-цен.
- $\beta' p_{t-1}$ — ошибка долгосрочного равновесия.
- $\alpha$ — скорость возврата к равновесию.
- $\Gamma_\ell$ — краткосрочная динамика.
- $u_t$ — новая инновация, то есть шок модели.

Размерности:

```math
\alpha \in \mathbb{R}^{n \times r}
```

```math
\beta \in \mathbb{R}^{n \times r}
```

```math
\Gamma_\ell \in \mathbb{R}^{n \times n}
```

```math
u_t \in \mathbb{R}^{n \times 1}
```

Здесь $r$ — ранг коинтеграции.

В коде после оценки VECM используются эти объекты:

```python
vecm_result.alpha
vecm_result.beta
vecm_result.gamma
vecm_result.resid
vecm_result.sigma_u
vecm_result.var_rep
```

## 4. Ковариация инноваций

Инновации $u_t$ имеют ковариационную матрицу:

```math
\Omega = E(u_t u_t')
```

На данных она оценивается по остаткам модели:

```math
\Omega
=
\frac{1}{T^*}
\sum_t
u_t u_t'
```

Размер:

```math
\Omega \in \mathbb{R}^{n \times n}
```

В коде:

```python
sigma = residual_covariance(vecm_result)
```

Важная идея: элементы вне диагонали обычно не равны нулю. Значит, шоки разных активов связаны между собой.

## 5. Долгосрочная матрица воздействия

GIS и Hasbrouck proxy используют долгосрочный эффект шоков.

Для этого нужна матрица $C$.

Сначала строятся ортогональные дополнения:

```math
\alpha_\perp' \alpha = 0
```

```math
\beta_\perp' \beta = 0
```

Если ранг коинтеграции равен $r$, то:

```math
\alpha_\perp \in \mathbb{R}^{n \times (n-r)}
```

```math
\beta_\perp \in \mathbb{R}^{n \times (n-r)}
```

Сумма краткосрочных матриц:

```math
\Gamma_\Sigma
=
\sum_{\ell=1}^{k}
\Gamma_\ell
```

Дальше:

```math
\bar{\Gamma}
=
I_n - \Gamma_\Sigma
```

Долгосрочная матрица:

```math
C
=
\beta_\perp
\left(
\alpha_\perp' \bar{\Gamma} \beta_\perp
\right)^{-1}
\alpha_\perp'
```

Размер:

```math
C \in \mathbb{R}^{n \times n}
```

В коде:

```python
long_run = long_run_impact_matrix(vecm_result, coint_rank=rank)
```

Простой смысл: $C$ показывает, какой постоянный эффект дает шок $u_t$.

## 6. GFEVD

GFEVD отвечает на вопрос:

> Чьи шоки объясняют ошибку прогноза каждого актива?

Для GFEVD нужны:

```math
A_1,\ldots,A_p
```

```math
\Omega
```

```math
H
```

Здесь $H$ — горизонт прогноза.

В коде VAR-матрицы берутся из VECM:

```python
A_mats = vecm_result.var_rep
```

Размер массива с VAR-матрицами:

```math
A_{\mathrm{mats}} \in \mathbb{R}^{p \times n \times n}
```

## 7. MA-матрицы для GFEVD

Сначала строятся матрицы реакции на шоки:

```math
\Psi_0 = I_n
```

Для $h=1,\ldots,H-1$:

```math
\Psi_h
=
\sum_{\ell=1}^{\min(h,p)}
A_\ell \Psi_{h-\ell}
```

Размер:

```math
\Psi \in \mathbb{R}^{H \times n \times n}
```

В коде:

```python
ma_mats = ma_representation(vecm_result.var_rep, horizon=horizon)
```

Простой смысл: $\Psi_h$ показывает, как шок сегодня влияет на цены через $h$ шагов.

## 8. Знаменатель GFEVD

Для актива $i$ полная дисперсия ошибки прогноза:

```math
D_i(H)
=
\sum_{h=0}^{H-1}
e_i' \Psi_h \Omega \Psi_h' e_i
```

Это число:

```math
D_i(H) \in \mathbb{R}
```

В коде:

```python
denominator += np.diag(ph @ sigma @ ph.T)
```

Простой смысл: это вся ошибка прогноза актива $i$.

## 9. Числитель GFEVD

Теперь берем только вклад шока актива $j$ в ошибку прогноза актива $i$:

```math
N_{ij}(H)
=
\sigma_{jj}^{-1}
\sum_{h=0}^{H-1}
\left(
e_i' \Psi_h \Omega e_j
\right)^2
```

В коде:

```python
cross = ph @ sigma
numerator[:, valid_diag] += (cross[:, valid_diag] ** 2) / sigma_diag[valid_diag]
```

Почему это то же самое:

```math
\left[\Psi_h \Omega\right]_{ij}
=
e_i' \Psi_h \Omega e_j
```

То есть элемент строки $i$ и столбца $j$ в матрице $\Psi_h \Omega$ — это нужная реакция.

## 10. Итоговая GFEVD

Сырая GFEVD:

```math
\theta_{ij}^{g}(H)
=
\frac{N_{ij}(H)}{D_i(H)}
```

Матрица:

```math
\Theta^g(H) \in \mathbb{R}^{n \times n}
```

Но строки сырой GFEVD могут не давать сумму 1. Поэтому делается нормировка:

```math
\widetilde{\theta}_{ij}^{g}(H)
=
\frac{
\theta_{ij}^{g}(H)
}{
\sum_{m=1}^{n}
\theta_{im}^{g}(H)
}
```

После нормировки:

```math
\sum_{j=1}^{n}
\widetilde{\theta}_{ij}^{g}(H)
=
1
```

В коде:

```python
raw = generalized_fevd(ma_mats, sigma)
normalized = row_normalize(raw)
```

Простой смысл: строка показывает, откуда пришла прогнозная дисперсия данного актива.

## 11. GIS

GIS отвечает на вопрос:

> Какой актив вносит большую долю в долгосрочную общую цену?

Для классического GIS нужно, чтобы был один общий стохастический тренд:

```math
r = n - 1
```

Для расчета нужны:

```math
C
```

```math
\Omega
```

## 12. Разложение ковариации для GIS

Сначала берем стандартные отклонения инноваций:

```math
D
=
\mathrm{diag}
\left(
\sqrt{\sigma_{11}},
\ldots,
\sqrt{\sigma_{nn}}
\right)
```

Потом строим корреляционную матрицу:

```math
R
=
D^{-1}
\Omega
D^{-1}
```

Разложение корреляционной матрицы:

```math
R
=
G \Lambda G'
```

Модифицированный корень ковариации:

```math
F_M
=
D G \Lambda^{1/2} G'
```

Проверка:

```math
F_M F_M'
=
\Omega
```

В коде:

```python
factor = modified_correlation_sqrt(sigma)
```

Простой смысл: $F_M$ превращает связанные шоки в ортогональные компоненты без Cholesky-порядка.

## 13. Итоговая GIS

Долгосрочный эффект ортогональных шоков:

```math
Q
=
C F_M
```

Столбец $j$:

```math
Q e_j
```

Это долгосрочный эффект шока $j$.

Квадрат длины этого эффекта:

```math
s_j
=
\left\| Q e_j \right\|_2^2
```

Общая сумма:

```math
S
=
\sum_{j=1}^{n}
s_j
```

GIS:

```math
GIS_j
=
\frac{s_j}{S}
```

В коде:

```python
effects = long_run @ factor
shares = column_shares(effects)
```

Функция `column_shares` считает:

```math
GIS_j
=
\frac{
\sum_i Q_{ij}^2
}{
\sum_j \sum_i Q_{ij}^2
}
```

Простой смысл: чем больше долгосрочный эффект столбца $j$, тем больше GIS актива $j$.

## 14. Hasbrouck proxy

Hasbrouck proxy тоже измеряет price discovery, но использует Cholesky-разложение.

Проблема: Cholesky зависит от порядка активов.

Поэтому считаются разные порядки, а потом берутся границы.

Для одного порядка $\pi$ строится матрица перестановки:

```math
P_\pi
```

Ковариация в этом порядке:

```math
\Omega_\pi
=
P_\pi \Omega P_\pi'
```

Cholesky-разложение:

```math
\Omega_\pi
=
L_\pi L_\pi'
```

В коде:

```python
sigma_ordered = sigma[np.ix_(order_idx, order_idx)]
chol = safe_cholesky(sigma_ordered)
```

Долгосрочный эффект Cholesky-шоков:

```math
Q_\pi
=
C P_\pi' L_\pi
```

В коде:

```python
effects_ordered = long_run[:, order_idx] @ chol
```

## 15. Доля Hasbrouck для одного порядка

Для шока $j$:

```math
s_j(\pi)
=
\left\| Q_\pi e_j \right\|_2^2
```

Доля:

```math
IS_j(\pi)
=
\frac{
s_j(\pi)
}{
\sum_{m=1}^{n}
s_m(\pi)
}
```

После расчета доли возвращаются к исходному порядку активов:

```python
shares[order_idx] = shares_ordered
```

Простой смысл: мы считаем вклад каждого актива при одном конкретном Cholesky-порядке.

## 16. Границы Hasbrouck proxy

Так как порядок влияет на результат, считаем несколько порядков.

Для каждого актива $j$:

```math
lower_j
=
\min_{\pi \in \Pi}
IS_j(\pi)
```

```math
upper_j
=
\max_{\pi \in \Pi}
IS_j(\pi)
```

Средняя точка интервала:

```math
midpoint_j
=
\frac{
lower_j + upper_j
}{2}
```

Среднее по порядкам:

```math
mean_j
=
\frac{1}{|\Pi|}
\sum_{\pi \in \Pi}
IS_j(\pi)
```

В коде:

```python
lower = order_shares.min(axis=0)
upper = order_shares.max(axis=0)
summary = pd.DataFrame({
    "lower": lower,
    "upper": upper,
    "midpoint": (lower + upper) / 2.0,
    "mean": order_shares.mean(axis=0),
    "std": order_shares.std(axis=0, ddof=0),
})
```

Простой смысл:

- `lower` — минимальная доля актива среди порядков.
- `upper` — максимальная доля актива среди порядков.
- `midpoint` — середина интервала.
- `mean` — средняя доля по порядкам.

## 17. Проверки

После нормировки строки GFEVD должны суммироваться в 1:

```math
\sum_{j=1}^{n}
\widetilde{\theta}_{ij}^{g}(H)
=
1
```

GIS должен суммироваться в 1:

```math
\sum_{j=1}^{n}
GIS_j
=
1
```

Hasbrouck-доли для каждого порядка должны суммироваться в 1:

```math
\sum_{j=1}^{n}
IS_j(\pi)
=
1
```

Средние Hasbrouck-доли тоже должны суммироваться в 1:

```math
\sum_{j=1}^{n}
mean_j
=
1
```

Эти проверки есть в:

```text
tests/test_metrics_smoke.py
```
