# Численные методы

Набор учебных реализаций численных методов на Python.

## Содержание

- `lab1` — решение СЛАУ (LU-разложение, метод прогонки, простые итерации и метод Зейделя), а также поиск собственных значений методами вращений Якоби и QR-разложения;
- `lab2` — решение нелинейного уравнения и системы нелинейных уравнений методами простой итерации и Ньютона;
- `lab3/lab3.1` — интерполяция функции ctg(x) многочленами Лагранжа и Ньютона (задание 3.1, вариант 4, оба набора узлов);
- `lab3/lab3.2` — естественный кубический сплайн по табличным данным (задание 3.2, вариант 4);
- `lab3/lab3.3` — приближение табличных данных многочленами МНК степеней 1, 2 и 3 (задание 3.3, вариант 4);
- `lab3/lab3.4` — первая и вторая производные табличной функции по трём узлам (задание 3.4, вариант 4);
- `lab3/lab3.5` — численное интегрирование пятью методами и уточнение Рунге–Ромберга (задание 3.5, вариант 4).

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

```bash
cd lab3/lab3.1
python interpolation.py
```

```bash
cd lab3/lab3.2
python cubic_spline.py
```

```bash
cd lab3/lab3.3
python least_squares.py
```

```bash
cd lab3/lab3.4
python numerical_differentiation.py
```

```bash
cd lab3/lab3.5
python numerical_integration.py
```

Исходные данные и параметры точности задаются в файлах `*_input.json`. Программы выводят результаты в консоль. Задания из `lab2`, а также `lab3.1`–`lab3.3` дополнительно сохраняют графики в PNG. В `lab3/lab3.1/interpolation_input.json` узлы и точка интерполяции заданы как рациональные доли π.
