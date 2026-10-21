# Численные методы

Набор учебных реализаций численных методов на Python.

## Содержание

- `lab1` — решение СЛАУ (LU-разложение, метод прогонки, простые итерации и метод Зейделя), а также поиск собственных значений методами вращений Якоби и QR-разложения;
- `lab2` — решение нелинейного уравнения и системы нелинейных уравнений методами простой итерации и Ньютона.

## Запуск

Для работы нужны Python 3, NumPy и Matplotlib:

```bash
python -m pip install numpy matplotlib
```

Скрипты следует запускать из каталога соответствующей лабораторной работы. Например:

```bash
cd lab1
python lu_method.py
python sweep_method.py
python iterative_methods.py
python jacobi_rotation.py
python qr_method.py
```

```bash
cd lab2
python nonlinear_equation.py
python nonlinear_system.py
```

Исходные данные и параметры точности задаются в файлах `*_input.json`. Программы выводят результаты в консоль. Задания из `lab2` дополнительно сохраняют графики в PNG.
