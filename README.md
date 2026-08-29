# soilactivity

Пакет для 3D-реконструкции объемной активности радионуклидов в почве по измерениям МАЭД. Вдохновлен лучшими практиками пакета `bssunfold` (Numba-ускорение, строгая индексация, оценка неопределенностей).

## Установка
```bash
pip install -e .[accel,dev]
```

## Быстрый старт
```python
from soilactivity import Unfolder
import pandas as pd
import numpy as np

# 1. Загрузка данных (пример)
data = pd.read_csv('measurements.csv') # колонки: x, y, z, dose_rate

# 2. Создание сетки
grid_x = np.linspace(0, 10, 20)
grid_y = np.linspace(0, 10, 20)
grid_z = np.linspace(0, 2, 5)

# 3. Инициализация и запуск
uf = Unfolder(method='mlem', iterations=50, tol=1e-4)
result = uf.unfold(
    data,
    (grid_x, grid_y, grid_z),
    attenuation_coeff=0.1
)

print(f"Сходимость за {result.solver_info['iterations']} итераций")
result.save_to_file('result_activity.npz')
```

## Факторы накопления (ANS-6.4.3)

Модуль `soilactivity.buildup` предоставляет факторы накопления B(E, x)
из стандарта ANSI/ANS-6.4.3-1991 (Trubey 1988, ORNL/RSIC-49) для
26 материалов (23 элемента + вода, воздух, бетон) на сетке
25 энергий (0.015–15 МэВ) × 16 глубин (0.5–40 mfp).

```python
from soilactivity import (
    get_buildup, gp_buildup_water, buildup_for_mixture, AVAILABLE_MATERIALS,
)

# Табличное значение с 2D log-log интерполяцией
B = get_buildup('Water', E_MeV=1.0, x_mfp=10.0)   # -> 26.1
B = get_buildup('Lead',  E_MeV=1.0, x_mfp=10.0)   # -> 3.37

# Векторизованный вызов
Es = np.array([0.1, 1.0, 10.0])
xs = np.array([1.0, 10.0, 40.0])
B = get_buildup('Water', Es, xs)

# GP-формула (Harima) для воды — быстрее таблиц, точность 1-3%
B = gp_buildup_water(E_MeV=1.0, x_mfp=10.0, response='air')   # exposure
B = gp_buildup_water(E_MeV=1.0, x_mfp=10.0, response='water')  # energy absorption

# Смесь/соединение через эквивалентный атомный номер Zeq
concrete = {'H':0.01,'O':0.529,'Mg':0.002,'Al':0.034,
            'Si':0.337,'Ca':0.044,'Fe':0.013}
B = buildup_for_mixture(concrete, E_MeV=1.0, x_mfp=10.0)

print(AVAILABLE_MATERIALS())
# ('Air', 'Aluminum', 'Argon', 'Beryllium', 'Boron', 'Calcium', 'Carbon',
#  'Concrete', 'Copper', 'Gadolinium', 'Iron', 'Lanthanum', 'Lead',
#  'Magnesium', 'Molybdenum', 'Nitrogen', 'Oxygen', 'Phosphorus',
#  'Potassium', 'Silicon', 'Sodium', 'Sulphur', 'Tin', 'Tungsten',
#  'Uranium', 'Water')
```

**Источники данных:**
- Trubey D. K. "New Gamma-Ray Buildup Factor Data for Point Kernel
  Calculations: ANS-6.4.3 Standard Reference Data." ORNL/RSIC-49 (1988).
  URL: https://inis.iaea.org/records/0arzw-ez976/files/20014493.pdf
- Harima Y. et al. "Validity of the Geometric-Progression Formula in
  Approximating Gamma-Ray Buildup Factors." Nucl. Sci. Eng. 94, 24-35 (1986).
- Olarinoye I. O. "EXABCal: A program for calculating photon exposure and
  energy absorption buildup factors." MethodsX 6, 1755-1763 (2019).

## Реконструкция ПДА по уравнению Фредгольма

Модуль `soilactivity.reconstructor` — высокоуровневый API для решения
обратной задачи: восстановление поверхностной плотности активности (ПДА)
из измерений мощности амбиентного дозного эквивалента (МАЭД) H*(10).

```python
from soilactivity import SadReconstructor
import numpy as np

ader_map = np.random.lognormal(0, 0.5, (50, 50))

recon = SadReconstructor(
    nx=50, ny=50, cell_size=10.0, height_m=1.0,
    radionuclide='Cs-137', dose_quantity='H_star_10',
)
result = recon.reconstruct(ader_map, alpha=1e-11)
print(f'Activity: {result.total_activity:.3e} Bq')
print(f'Gini: {result.info["gini_sad"]:.3f}')
```

| Параметр | Описание | По умолчанию |
|---|---|---|
| `nx`, `ny` | Размер растра | — |
| `cell_size` | Размер ячейки, м (рек. <= 3 м) | — |
| `height_m` | Высота детектора, м | 1.0 |
| `radionuclide` | Ключ из KERMA_CONSTANTS | `'Cs-137'` |
| `dose_quantity` | `H_star_10`, `K_air`, `D_air`, `X` | `'H_star_10'` |
| `buildings` | Барьеры `[{x,y,width,height}]` | `None` |

### Доступные радионуклиды

Cs-137, Cs-134, Co-60, Co-58, Eu-152, Eu-154, I-131, Ba-140, Zr-95,
Nb-95, Ru-103, Ru-106, Ce-141, Ce-144, La-140, Mn-54, Fe-59,
Zn-65, Sb-124, Am-241, Sr-90, Y-90.

## Примеры (examples/)

### Синтетические

| Ноутбук | Описание |
|---|---|
| `example00.ipynb` | Фурье-свертка для восстановления карты активности |
| `example01.ipynb` | 3D Cs-137: MLEM, Tikhонова, matplotlib 3D, Plotly |
| `example02.ipynb` | SRTM-рельеф, Sr-90, Plotly 3D, интерполяция МАЭД |

### Реальные данные

| Ноутбук | Описание | Источник |
|---|---|---|
| `example03_chernobyl.ipynb` | ЧЗО: Cs-137/Sr-90, Фредгольм, Лоренц, Джини | Kashparov (2018, 2020) |
| `example04_semei.ipynb` | СИП: Cs-137/Sr-90/Co-60, 3 площадки | OSTI, PMC, IAEA |
| `example05_co60.ipynb` | Промышленный Co-60: барьеры, теневой эффект | Al Tuwaitha, Hanford |

### bssunfold — 66 методов

`examples/bssunfold_methods/` (01–66): TSVD, MLEM, Tikhonov, Landweber,
CGLS, Kaczmarz, SART, FISTA, Gravel, MAXED, SAND-II, BUNKI, Bayes,
LMfit, SciPy, Mystic, Genetic, CPLEX, QUBO, ODL, zfit и др.

### Анализ и интерполяция

| Ноутбук | Описание |
|---|---|
| `example06_interpretation.ipynb` | Автоподбор интерполяции, анализ влияния точек, разреженные результаты |

## Пространственная интерполяция и анализ чувствительности

Модуль `soilactivity.spatial_interpolation` обеспечивает единый интерфейс для
14 методов 2D-интерполяции, автоматический подбор лучшего метода через
кросс-валидацию, анализ влияния точек измерений на результат и интерполяцию
разреженных реконструкций с оценкой неопределённости.

### Доступные методы

| Метод | Класс | Описание | Зависимость |
|---|---|---|---|
| `rbf_tps` | RBF | Тонкая пластинка (smooth, по умолчанию) | scipy |
| `rbf_linear` | RBF | Линейное RBF | scipy |
| `rbf_cubic` | RBF | Кубическое RBF | scipy |
| `rbf_gaussian` | RBF | Гауссово RBF | scipy |
| `nearest` | Delaunay | Ближайший сосед (быстро, без сглаживания) | scipy |
| `linear_delaunay` | Delaunay | Линейная триангуляция | scipy |
| `cubic_delaunay` | Delaunay | Clough-Tocher кубическое (C1 гладкое) | scipy |
| `idw` | Детермин. | Обратные расстояния (power=2, k=12) | scipy |
| `barnes` | Метео | Последовательные поправки Барнса | numpy |
| `cressman` | Метео | Схема Крессмана | numpy |
| `kriging` | Геостат. | Обычный кригинг | pykrige |
| `gp_rbf` | GP | Гауссовский процесс, RBF ядро | scikit-learn |
| `gp_matern32` | GP | Гауссовский процесс, Matern 3/2 | scikit-learn |
| `gp_matern52` | GP | Гауссовский процесс, Matern 5/2 | scikit-learn |

### Interpolator2D — единый интерфейс

```python
from soilactivity import Interpolator2D

interp = Interpolator2D(method='rbf_tps', smoothing=0.1)
interp.fit(x, y, z)                     # обучить на точках измерений
zi = interp.predict(xi, yi)              # интерполяция в произвольные точки
Z, XI, YI = interp.predict_grid(xi, yi)  # интерполяция на регулярную сетку
std = interp.uncertainty(XI, YI)         # неопределённость (GP, Kriging)
print(interp.get_info())                 # справка о методе
```

### InterpolationAutoSelector — автоматический подбор

```python
from soilactivity import InterpolationAutoSelector

selector = InterpolationAutoSelector(
    candidates=['rbf_tps', 'linear_delaunay', 'idw', 'barnes', 'cressman'],
    cv_folds=5,
    metrics=['rmse', 'mae', 'r2'],
)
selector.fit(x, y, z)
result = selector.select()           # {'best_method': 'rbf_tps', 'best_score': 0.042, ...}
print(selector.get_recommendation())  # текстовая рекомендация
ranking = selector.get_ranking()     # все методы отсортированы по RMSE
fig, ax = selector.plot_comparison() # столбчатая диаграмма RMSE
```

Выбор: минимальный RMSE, при равенстве — максимальный R². Для N < 30
автоматически используется leave-one-out вместо k-fold.

### MeasurementSensitivityAnalyzer — анализ влияния точек

Аналог `bssunfold.unfold_interpret` и подхода `pyoptexplain`.
Определяет, какие точки измерений больше всего влияют на результат
интерполяции (и, следовательно, на реконструкцию активности).

```python
from soilactivity import MeasurementSensitivityAnalyzer

analyzer = MeasurementSensitivityAnalyzer()
analyzer.fit(x, y, z, method='rbf_tps')

# Leave-one-out: удалить каждую точку, пересчитать поле
loo = analyzer.sensitivity_leave_one_out()

# Perturbation: Perturb z[i] на delta_frac, измерить изменение поля
pert = analyzer.sensitivity_perturbation(delta_frac=0.1)

# Результаты
ranking = analyzer.ranking()                  # сортировка по max_influence
critical = analyzer.critical_points(90)         # top-10% критических точек

# Карта суммарного влияния
influence = analyzer.influence_map(xi, yi)
fig, ax = analyzer.plot_influence(xi, yi)     # heatmap + точки измерений
```

Каждая точка в результате содержит:
`point_index`, `x`, `y`, `z`, `max_influence`, `mean_influence`,
`influence_area_km2`.

### SparseResultInterpolator — интерполяция разреженных результатов

Когда Фредголm-реконструкция даёт результат в нескольких точках,
этот класс интерполирует их на плотную сетку с оценкой неопределённости.

```python
from soilactivity import SparseResultInterpolator

spi = SparseResultInterpolator(method='gp_rbf', uncertainty_threshold=0.3)
spi.fit_sparse(reconstructed_points, values, uncertainty=unc)
result = spi.interpolate_to_grid(xmin, xmax, ymin, ymax, nx=50, ny=50)

print(result.method_used)       # 'gp_rbf'
print(result.n_input_points)    # 8
print(result.coverage)          # 0.72 — доля ячеек с confidence
print(result.interpolated.shape)  # (50, 50)
print(result.uncertainty.shape)    # (50, 50) или None
print(result.confidence_mask.shape)  # (50, 50) bool
```

Рекомендуется использовать GP-методы (`gp_rbf`, `gp_matern52`) для
автоматической оценки неопределённости. Если предоставлены `uncertainty`,
они используются как prior noise в GP.

### Standalone-функции

```python
from soilactivity import idw_interpolate, barnes_interpolate, cressman_interpolate

zi = idw_interpolate(x, y, z, xi, yi, power=2, max_neighbors=12)
zi = barnes_interpolate(x, y, z, xi, yi, kappa=5.0, iterations=2)
zi = cressman_interpolate(x, y, z, xi, yi, radius=5.0)
```

### Зависимости

| Пакет | Обязательный | Для методов |
|---|---|---|
| numpy | Да | все |
| scipy | Да | RBF, Delaunay, IDW |
| scikit-learn | Нет | `gp_rbf`, `gp_matern32`, `gp_matern52` |
| pykrige | Нет | `kriging` |
| matplotlib | Нет | графики (`plot_influence`, `plot_comparison`) |

## Пространственная статистика

```python
from soilactivity import lorenz_curve, lorenz_gini_coefficient

gini = lorenz_gini_coefficient(activity_map)
lcx, lcy = lorenz_curve(activity_map)
cc = lorenz_compactness_ratio(sad_map, ader_map)
```

## Лицензия

MIT
