"""Численное интегрирование."""

import json
import math
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "integration_input.json"


def function(x):
    """Подынтегральная функция (3x + 4) / (2x + 7)."""
    return (3 * x + 4) / (2 * x + 7)


def exact_integral(left, right):
    """Интеграл от первообразной 3x/2 - 13/4 ln|2x + 7|."""
    return 1.5 * (right - left) - 3.25 * math.log(
        abs((2 * right + 7) / (2 * left + 7))
    )


def read_data(filename):
    with filename.open(encoding="utf-8") as file:
        config = json.load(file)

    left, right = map(float, config["interval"])
    h1 = float(config["h1"])
    h2 = float(config["h2"])
    precision = int(config["output_precision"])
    if not all(map(math.isfinite, (left, right, h1, h2))):
        raise ValueError("границы и шаги должны быть конечными числами")
    if left >= right or h1 <= h2 or h2 <= 0:
        raise ValueError("необходимо left < right и 0 < h2 < h1")
    if left <= -3.5 <= right:
        raise ValueError("интервал содержит полюс функции x = -3.5")
    if not 1 <= precision <= 16:
        raise ValueError("output_precision должен быть от 1 до 16")

    counts = []
    for step in (h1, h2):
        count_real = (right - left) / step
        count = round(count_real)
        if count <= 0 or not math.isclose(count_real, count, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("каждый шаг должен точно разбивать интервал")
        counts.append(count)
    return left, right, h1, h2, counts, precision


def grid_points(left, right, count):
    return np.linspace(left, right, count + 1)


def left_rectangles(left, right, count):
    grid = grid_points(left, right, count)
    return (right - left) / count * np.sum(function(grid[:-1]))


def right_rectangles(left, right, count):
    grid = grid_points(left, right, count)
    return (right - left) / count * np.sum(function(grid[1:]))


def midpoint_rectangles(left, right, count):
    grid = grid_points(left, right, count)
    midpoints = (grid[:-1] + grid[1:]) / 2
    return (right - left) / count * np.sum(function(midpoints))


def trapezoids(left, right, count):
    grid = grid_points(left, right, count)
    values = function(grid)
    return (right - left) / count * (
        (values[0] + values[-1]) / 2 + np.sum(values[1:-1])
    )


def simpson(left, right, count):
    if count % 2:
        raise ValueError("метод Симпсона требует чётного числа промежутков")
    grid = grid_points(left, right, count)
    values = function(grid)
    return (right - left) / (3 * count) * (
        values[0] + values[-1]
        + 4 * np.sum(values[1:-1:2])
        + 2 * np.sum(values[2:-1:2])
    )


METHODS = (
    ("Левые прямоугольники", 1, left_rectangles),
    ("Правые прямоугольники", 1, right_rectangles),
    ("Средние прямоугольники", 2, midpoint_rectangles),
    ("Трапеции", 2, trapezoids),
    ("Симпсон", 4, simpson),
)


def runge_romberg(coarse, fine, ratio, order):
    """Уточнить I(h2) и оценить его ведущую погрешность."""
    correction = (fine - coarse) / (ratio ** order - 1)
    return fine + correction, abs(correction)


def main():
    left, right, h1, h2, (n1, n2), precision = read_data(INPUT_FILE)
    if n1 % 2 or n2 % 2:
        raise ValueError("для метода Симпсона оба числа промежутков должны быть чётными")
    exact = exact_integral(left, right)
    ratio = h1 / h2
    results = []
    for name, order, method in METHODS:
        coarse = float(method(left, right, n1))
        fine = float(method(left, right, n2))
        refined, error_estimate = runge_romberg(coarse, fine, ratio, order)
        results.append((name, order, coarse, fine, refined, error_estimate))

    print(f"f(x) = (3x + 4) / (2x + 7), интервал [{left:g}; {right:g}]")
    print(f"h1 = {h1:g} (n1 = {n1}), h2 = {h2:g} (n2 = {n2})")
    print("Уточнение Рунге–Ромберга: I_RR = I(h2) + [I(h2) - I(h1)] / [(h1/h2)^p - 1]")
    print("Оценка погрешности I(h2): |I_RR - I(h2)|")
    print("\nРезультаты численного интегрирования:")
    print(
        f"{'Метод':<24} {'I(h1)':>16} {'I(h2)':>16} "
        f"{'p':>3} {'I_RR':>16} {'Оценка |E_h2|':>18}"
    )
    for name, order, coarse, fine, refined, estimate in results:
        print(
            f"{name:<24} {coarse:>16.{precision}g} {fine:>16.{precision}g} "
            f"{order:>3} {refined:>16.{precision}g} {estimate:>18.{precision}g}"
        )

    print("\nПервообразная: F(x) = 3x/2 - (13/4) ln|2x + 7|")
    print(f"Точное значение интеграла: {exact:.{precision}g}")
    print("Абсолютные погрешности:")
    print(f"{'Метод':<24} {'|E_h1|':>16} {'|E_h2|':>16} {'|E_RR|':>16}")
    for name, _, coarse, fine, refined, _ in results:
        print(
            f"{name:<24} {abs(coarse - exact):>16.{precision}g} "
            f"{abs(fine - exact):>16.{precision}g} "
            f"{abs(refined - exact):>16.{precision}g}"
        )

    best_fine = min(results, key=lambda row: abs(row[3] - exact))
    best_refined = min(results, key=lambda row: abs(row[4] - exact))
    print(f"\nНаиболее точный результат с шагом h2: {best_fine[0]}")
    print(f"Наиболее точный результат после уточнения: {best_refined[0]}")


if __name__ == "__main__":
    main()
