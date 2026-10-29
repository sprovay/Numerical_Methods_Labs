"""Кубический сплайн."""

import json
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "spline_input.json"
sys.path.insert(0, str(BASE_DIR.parents[1] / "lab1"))

from sweep_method import sweep_method


def read_data(filename):
    """Прочитать таблицу и проверить область построения сплайна."""
    with filename.open(encoding="utf-8") as file:
        config = json.load(file)

    nodes = np.asarray(config["nodes"], dtype=float)
    values = np.asarray(config["values"], dtype=float)
    x_star = float(config["x_star"])
    precision = int(config["output_precision"])
    if nodes.ndim != 1 or len(nodes) < 2 or values.shape != nodes.shape:
        raise ValueError("нужны не менее двух узлов и по одному значению для каждого узла")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(values)):
        raise ValueError("узлы и значения должны быть конечными числами")
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("узлы должны строго возрастать")
    if not np.isfinite(x_star) or not nodes[0] <= x_star <= nodes[-1]:
        raise ValueError("x* должен лежать между крайними узлами")
    if not 1 <= precision <= 16:
        raise ValueError("output_precision должен быть от 1 до 16")
    return config, nodes, values, x_star, precision


def natural_cubic_spline(nodes, values):
    """Вернуть [a_i, b_i, c_i, d_i] и вторые производные в узлах."""
    steps = np.diff(nodes)
    slopes = np.diff(values) / steps
    second_derivatives = np.zeros(len(nodes))

    if len(nodes) > 2:
        diagonal = 2 * (steps[:-1] + steps[1:])
        off_diagonal = steps[1:-1]
        right_side = 6 * (slopes[1:] - slopes[:-1])
        second_derivatives[1:-1], _, _ = sweep_method(
            off_diagonal, diagonal, off_diagonal, right_side
        )

    a = values[:-1].copy()
    b = slopes - steps * (
        2 * second_derivatives[:-1] + second_derivatives[1:]
    ) / 6
    c = second_derivatives[:-1] / 2
    d = (second_derivatives[1:] - second_derivatives[:-1]) / (6 * steps)
    return np.column_stack((a, b, c, d)), second_derivatives


def interval_index(nodes, x):
    """Найти участок."""
    argument = np.asarray(x, dtype=float)
    if np.any((argument < nodes[0]) | (argument > nodes[-1])):
        raise ValueError("точка находится вне области сплайна")
    return np.clip(np.searchsorted(nodes, argument, side="right") - 1, 0, len(nodes) - 2)


def spline_value(nodes, coefficients, x):
    """Вычислить сплайн в точке или массиве точек."""
    argument = np.asarray(x, dtype=float)
    indices = interval_index(nodes, argument)
    a, b, c, d = np.moveaxis(coefficients[indices], -1, 0)
    shift = argument - nodes[indices]
    result = a + shift * (b + shift * (c + shift * d))
    return float(result) if result.ndim == 0 else result


def polynomial_text(coefficients, node, precision):
    """Представить участок сплайна по степеням (x - x_i)."""
    text = f"{coefficients[0]:.{precision}g}"
    for degree, coefficient in enumerate(coefficients[1:], start=1):
        if coefficient == 0:
            continue
        power = "" if degree == 1 else f"^{degree}"
        sign = "+" if coefficient >= 0 else "-"
        text += f" {sign} {abs(coefficient):.{precision}g}(x - {node:g}){power}"
    return text


def create_plot(nodes, values, coefficients, x_star, spline_at_star, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 6))
    colors = plt.get_cmap("tab10")
    for i in range(len(coefficients)):
        grid = np.linspace(nodes[i], nodes[i + 1], 150)
        axes.plot(
            grid, spline_value(nodes, coefficients, grid),
            color=colors(i), linewidth=2.2,
            label=f"S{i}(x), [{nodes[i]:g}; {nodes[i + 1]:g}]",
        )
    axes.scatter(nodes, values, color="black", zorder=4, label="Исходные узлы")
    axes.scatter(
        [x_star], [spline_at_star], color="red", marker="*", s=180,
        zorder=5, label="(x*, S(x*))",
    )
    axes.axvline(x_star, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    axes.set(xlabel="x", ylabel="S(x)", title="Естественный кубический сплайн, вариант 4")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    config, nodes, values, x_star, precision = read_data(INPUT_FILE)
    coefficients, _ = natural_cubic_spline(nodes, values)
    segment = int(interval_index(nodes, x_star))
    result = spline_value(nodes, coefficients, x_star)
    node_results = spline_value(nodes, coefficients, nodes)
    
    print(f"Граничные условия: S''(x0) = S''(x{len(nodes) - 1}) = 0")
    print("Исходные узлы и значения функции:")
    for i, (node, value) in enumerate(zip(nodes, values)):
        print(f"  x{i} = {node:.{precision}g}; y{i} = {value:.{precision}g}")

    print("\nS_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3")
    print("Коэффициенты и интервалы действия:")
    print(f"{'i':>2} {'интервал':>14} {'a_i':>16} {'b_i':>16} {'c_i':>16} {'d_i':>16}")
    for i, (a, b, c, d) in enumerate(coefficients):
        interval = f"[{nodes[i]:g}; {nodes[i + 1]:g}]"
        print(
            f"{i:>2} {interval:>14} "
            f"{a:>16.{precision}g} {b:>16.{precision}g} "
            f"{c:>16.{precision}g} {d:>16.{precision}g}"
        )
        print(f"   S{i}(x) = {polynomial_text((a, b, c, d), nodes[i], precision)}")
        print()

    print(f"\nx* = {x_star:.{precision}g}")
    print(
        f"Участок №{segment + 1} (i = {segment}): "
        f"[{nodes[segment]:g}; {nodes[segment + 1]:g}]"
    )
    print(f"S(x*) = {result:.{precision}g}")
    print("\nПроверка прохождения через узлы:")
    for i, (expected, actual) in enumerate(zip(values, node_results)):
        print(
            f"  x{i}: S(x{i}) = {actual:.{precision}g}, "
            f"y{i} = {expected:.{precision}g}, "
            f"|S(x{i}) - y{i}| = {abs(actual - expected):.{precision}g}"
        )
    print("\nПроверка граничных условий:")
    left_curvature = 2 * coefficients[0, 2]
    last_step = nodes[-1] - nodes[-2]
    right_curvature = 2 * coefficients[-1, 2] + 6 * coefficients[-1, 3] * last_step
    print(f"S''(x0) = {left_curvature:.{precision}g}")
    print(f"S''(x{len(nodes) - 1}) = {right_curvature:.{precision}g}")

    plot_path = BASE_DIR / config["plot_file"]
    create_plot(nodes, values, coefficients, x_star, result, plot_path)
    print(f"График: {plot_path.name}")


if __name__ == "__main__":
    main()
