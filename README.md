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
