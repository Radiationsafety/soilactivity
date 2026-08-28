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

## Пространственная статистика

```python
from soilactivity import lorenz_curve, lorenz_gini_coefficient

gini = lorenz_gini_coefficient(activity_map)
lcx, lcy = lorenz_curve(activity_map)
cc = lorenz_compactness_ratio(sad_map, ader_map)
```

## Лицензия

MIT
