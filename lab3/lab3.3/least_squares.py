"""Метод наименьших квадратов."""

import json
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "least_squares_input.json"
DEGREES = (1, 2, 3)
sys.path.insert(0, str(BASE_DIR.parents[1] / "lab1"))

from lu_method import lu_decomposition, solve_lu


def read_data(filename):
    """Прочитать табличные данные и проверить применимость МНК."""
    with filename.open(encoding="utf-8") as file:
        config = json.load(file)

    nodes = np.asarray(config["nodes"], dtype=float)
    values = np.asarray(config["values"], dtype=float)
    precision = int(config["output_precision"])
    if nodes.ndim != 1 or values.shape != nodes.shape or len(nodes) < 4:
        raise ValueError("нужны не менее четырёх пар (x_i, y_i)")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(values)):
        raise ValueError("исходные данные должны быть конечными числами")
    if len(np.unique(nodes)) < 4:
        raise ValueError("для многочлена третьей степени нужны четыре различных x_i")
    if not 1 <= precision <= 16:
        raise ValueError("output_precision должен быть от 1 до 16")
    return config, nodes, values, precision


def fit_least_squares(nodes, values, degree):
    """Решить нормальную систему МНК через LU-разложение."""
    design = np.vander(nodes, N=degree + 1, increasing=True)
    normal_matrix = design.T @ design
    normal_right = design.T @ values
    p, l, u, _ = lu_decomposition(normal_matrix)
    return solve_lu(p, l, u, normal_right)


def polynomial_value(coefficients, x):
    """Вычислить многочлен по схеме Горнера."""
    argument = np.asarray(x, dtype=float)
    result = np.zeros_like(argument, dtype=float)
    for coefficient in coefficients[::-1]:
        result = result * argument + coefficient
    return float(result) if result.ndim == 0 else result


def polynomial_text(coefficients, precision):
    """Показать многочлен в явном виде по степеням x."""
    result = f"{coefficients[0]:.{precision}g}"
    for power, coefficient in enumerate(coefficients[1:], start=1):
        sign = "+" if coefficient >= 0 else "-"
        variable = "x" if power == 1 else f"x^{power}"
        result += f" {sign} {abs(coefficient):.{precision}g}{variable}"
    return result


def errors(nodes, values, coefficients):
    """Сумма и среднее квадратов ошибок по всем исходным точкам."""
    residuals = polynomial_value(coefficients, nodes) - values
    squared = residuals * residuals
    return float(np.sum(squared)), float(np.mean(squared))


def create_plot(nodes, values, coefficients_by_degree, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(np.min(nodes), np.max(nodes), 600)
    figure, axes = plt.subplots(figsize=(9, 6))
    axes.scatter(nodes, values, color="black", s=55, zorder=4, label="Исходные данные")
    styles = {1: "--", 2: "-.", 3: "-"}
    for degree, coefficients in coefficients_by_degree.items():
        axes.plot(
            grid, polynomial_value(coefficients, grid), styles[degree],
            linewidth=2, label=f"P{degree}(x)",
        )
    axes.set(xlabel="x", ylabel="y", title="Метод наименьших квадратов, вариант 4")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    config, nodes, values, precision = read_data(INPUT_FILE)
    coefficients_by_degree = {}
    mse_by_degree = {}

    print("Исходные точки:")
    for i, (node, value) in enumerate(zip(nodes, values)):
        print(f"  i = {i}: x_i = {node:.{precision}g}, y_i = {value:.{precision}g}")
    print(f"MSE = SSE / {len(nodes)} (по всем исходным точкам)")

    for degree in DEGREES:
        coefficients = fit_least_squares(nodes, values, degree)
        sse, mse = errors(nodes, values, coefficients)
        coefficients_by_degree[degree] = coefficients
        mse_by_degree[degree] = mse

        print(f"\nМногочлен степени {degree}:")
        for power, coefficient in enumerate(coefficients):
            print(f"  a{power} = {coefficient:.{precision}g}")
        print(f"P{degree}(x) = {polynomial_text(coefficients, precision)}")
        print(f"SSE{degree} = {sse:.{precision}g}")
        print(f"MSE{degree} = {mse:.{precision}g}")

    order = sorted(DEGREES, key=mse_by_degree.get)
    comparison = " ≤ ".join(f"MSE{degree}" for degree in order)
    best = order[0]
    print(f"\nСравнение ошибок: {comparison}")
    print(f"Лучше всего данные приближает P{best}(x): MSE{best} = {mse_by_degree[best]:.{precision}g}")

    plot_path = BASE_DIR / config["plot_file"]
    create_plot(nodes, values, coefficients_by_degree, plot_path)
    print(f"График: {plot_path.name}")


if __name__ == "__main__":
    main()
