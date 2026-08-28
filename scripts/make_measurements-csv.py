import numpy as np
import pandas as pd

# Количество случайных точек
n_points = 2000  # Можно изменить на любое количество

# Генерация случайных координат в заданных диапазонах
x = np.random.uniform(0, 10, n_points)
y = np.random.uniform(0, 10, n_points)
z = np.random.uniform(0, 2, n_points)

# Генерация случайных значений dose_rate
# Например, нормальное распределение со средним 50 и стандартным отклонением 15
dose_rate = np.random.normal(loc=50, scale=15, size=n_points)
# Ограничиваем значения, чтобы не было отрицательных
dose_rate = np.maximum(dose_rate, 0)

# Создание DataFrame
data = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z,
    'dose_rate': dose_rate
})

# Сохранение в CSV файл
data.to_csv('measurements.csv', index=False)

print(f"Создан файл с {len(data)} случайными измерениями")
print("\nПервые 10 строк:")
print(data.head(10))
print(f"\nСтатистика:")
print(data.describe())