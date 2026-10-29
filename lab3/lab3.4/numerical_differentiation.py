"""Численное дифференцирование."""

import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "differentiation_input.json"


def read_data(filename):
    """Прочитать таблицу и проверить узлы и точку вычисления."""
    with filename.open(encoding="utf-8") as file:
        config = json.load(file)

    nodes = np.asarray(config["nodes"], dtype=float)
    values = np.asarray(config["values"], dtype=float)
    x_star = float(config["x_star"])
    precision = int(config["output_precision"])
    if nodes.ndim != 1 or len(nodes) < 3 or values.shape != nodes.shape:
        raise ValueError("нужны не менее трёх пар (x_i, y_i)")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(values)):
        raise ValueError("узлы и значения должны быть конечными числами")
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("узлы должны строго возрастать")
    if not np.isfinite(x_star) or not nodes[0] <= x_star <= nodes[-1]:
        raise ValueError("точка x* должна лежать внутри таблицы")
    if not 1 <= precision <= 16:
        raise ValueError("output_precision должен быть от 1 до 16")
    return nodes, values, x_star, precision


def select_stencil(nodes, x_star):
    """Взять три ближайших узла и вернуть их индексы по возрастанию x."""
    nearest = np.argsort(np.abs(nodes - x_star), kind="stable")[:3]
    return np.sort(nearest)


def quadratic_derivatives(nodes, values, x_star):
    """Производные квадратичного интерполянта Ньютона в точке x*."""
    x0, x1, x2 = nodes
    y0, y1, y2 = values
    first_difference = (y1 - y0) / (x1 - x0)
    second_difference = (y2 - y1) / (x2 - x1)
    divided_difference = (second_difference - first_difference) / (x2 - x0)

    first_derivative = first_difference + divided_difference * (2 * x_star - x0 - x1)
    second_derivative = 2 * divided_difference
    return first_derivative, second_derivative


def main():
    nodes, values, x_star, precision = read_data(INPUT_FILE)
    selected = select_stencil(nodes, x_star)
    used_nodes = nodes[selected]
    used_values = values[selected]
    first, second = quadratic_derivatives(used_nodes, used_values, x_star)

    print("Исходная таблица значений функции:")
    for i, (node, value) in enumerate(zip(nodes, values)):
        print(f"  i = {i}: x_i = {node:.{precision}g}, y_i = {value:.{precision}g}")
    print(f"\nТочка вычисления: x* = {x_star:.{precision}g}")
    print("Использованные узлы:")
    for i in selected:
        print(f"  i = {i}: x_i = {nodes[i]:.{precision}g}, y_i = {values[i]:.{precision}g}")
    print(f"\nf'(x*) ≈ {first:.{precision}g}")
    print(f"f''(x*) ≈ {second:.{precision}g}")


if __name__ == "__main__":
    main()
