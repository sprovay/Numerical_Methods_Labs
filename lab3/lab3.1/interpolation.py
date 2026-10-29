"""Интерполяционные многочлены Лагранжа и Ньютона."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "interpolation_input.json"


def cotangent(x):
    """Исходная функция варианта 4."""
    cosine = np.cos(x)
    return np.where(np.abs(cosine) < 1e-15, 0.0, cosine / np.sin(x))


def divided_differences(nodes, values):
    """Вернуть верхнетреугольную таблицу разделённых разностей."""
    n = len(nodes)
    table = np.full((n, n), np.nan)
    table[:, 0] = values
    for order in range(1, n):
        for row in range(n - order):
            table[row, order] = (
                table[row + 1, order - 1] - table[row, order - 1]
            ) / (nodes[row + order] - nodes[row])
    return table


def lagrange_value(nodes, values, x):
    """Вычислить многочлен Лагранжа в одной точке или на массиве точек."""
    argument = np.asarray(x, dtype=float)
    result = np.zeros_like(argument, dtype=float)
    for i, value in enumerate(values):
        term = np.ones_like(argument, dtype=float)
        for j, node in enumerate(nodes):
            if i != j:
                term *= (argument - node) / (nodes[i] - node)
        result += value * term
    return float(result) if result.ndim == 0 else result


def newton_value(nodes, coefficients, x):
    """Вычислить многочлен Ньютона вложенным умножением."""
    argument = np.asarray(x, dtype=float)
    result = np.full_like(argument, coefficients[-1], dtype=float)
    for i in range(len(coefficients) - 2, -1, -1):
        result = coefficients[i] + (argument - nodes[i]) * result
    return float(result) if result.ndim == 0 else result


def pi_label(fraction):
    """Представить рациональную долю π для печати многочленов."""
    numerator = "" if fraction.numerator == 1 else str(fraction.numerator)
    denominator = "" if fraction.denominator == 1 else f"/{fraction.denominator}"
    return f"{numerator}π{denominator}"


def polynomial_text(terms):
    """Собрать сумму с явными числовыми коэффициентами."""
    result = []
    for coefficient, factors in terms:
        if coefficient == 0:
            continue
        sign = "-" if coefficient < 0 else "+"
        body = f"{abs(coefficient):.12g}" + "".join(factors)
        if not result:
            result.append(f"-{body}" if coefficient < 0 else body)
        else:
            result.append(f" {sign} {body}")
    return "".join(result) if result else "0"


def lagrange_polynomial_text(nodes, values, labels):
    terms = []
    for i, value in enumerate(values):
        denominator = np.prod([nodes[i] - nodes[j] for j in range(len(nodes)) if j != i])
        factors = [f"(x - {labels[j]})" for j in range(len(nodes)) if j != i]
        terms.append((value / denominator, factors))
    return polynomial_text(terms)


def newton_polynomial_text(coefficients, labels):
    return polynomial_text([
        (coefficient, [f"(x - {label})" for label in labels[:order]])
        for order, coefficient in enumerate(coefficients)
    ])


def print_divided_differences(table):
    print("Таблица разделённых разностей:")
    headings = ["i"] + [f"порядок {order}" for order in range(table.shape[0])]
    print(f"{headings[0]:>3}" + "".join(f"{heading:>20}" for heading in headings[1:]))
    for i, row in enumerate(table):
        cells = [f"{value:.12g}" if np.isfinite(value) else "—" for value in row]
        print(f"{i:>3}" + "".join(f"{cell:>20}" for cell in cells))


def create_plot(nodes, values, coefficients, x_star, x_star_label, labels, case, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(nodes[0], nodes[-1], 600)
    figure, axes = plt.subplots(figsize=(9, 6))
    axes.plot(grid, cotangent(grid), label="f(x) = ctg(x)", linewidth=2.2)
    axes.plot(grid, lagrange_value(nodes, values, grid), "--", label="Лагранж")
    axes.plot(
        grid, newton_value(nodes, coefficients, grid), ":", linewidth=2.5,
        label="Ньютон",
    )
    axes.scatter(nodes, values, color="black", zorder=4, label="Узлы интерполяции")
    axes.scatter(
        [x_star], [cotangent(x_star)], marker="*", s=180, color="red",
        zorder=5, label=f"x* = {x_star_label}",
    )
    axes.axvline(x_star, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    for node, value, label in zip(nodes, values, labels):
        axes.annotate(label, (node, value), xytext=(4, 8), textcoords="offset points")
    axes.set(xlabel="x", ylabel="y", title=f"Интерполяция ctg(x), вариант 4{case}")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def read_fraction(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(part) is not int for part in value)
    ):
        raise ValueError("доля π должна быть парой целых чисел [числитель, знаменатель]")
    return Fraction(*value)


def main():
    with INPUT_FILE.open(encoding="utf-8") as file:
        config = json.load(file)
    x_star_fraction = read_fraction(config["x_star_pi_fraction"])
    x_star = float(x_star_fraction) * np.pi
    precision = int(config["output_precision"])
    if not 1 <= precision <= 16:
        raise ValueError("output_precision должен быть от 1 до 16")
    if not 0 < x_star < np.pi:
        raise ValueError("x* должен лежать внутри (0, π)")

    print(f"f(x) = ctg(x), x* = {pi_label(x_star_fraction)} = {x_star:.{precision}g}")
    for case in config["cases"]:
        fractions = [read_fraction(item) for item in case["nodes_pi_fractions"]]
        if len(fractions) != 4 or fractions != sorted(set(fractions)):
            raise ValueError("для каждого случая нужны четыре различных возрастающих узла")
        if not all(0 < fraction < 1 for fraction in fractions):
            raise ValueError("узлы ctg(x) должны лежать внутри (0, π)")
        nodes = np.array([float(fraction) * np.pi for fraction in fractions])
        if not nodes[0] <= x_star <= nodes[-1]:
            raise ValueError("x* должен лежать между крайними узлами")
        values = cotangent(nodes)
        table = divided_differences(nodes, values)
        coefficients = table[0]
        labels = [pi_label(fraction) for fraction in fractions]
        lagrange_result = lagrange_value(nodes, values, x_star)
        newton_result = newton_value(nodes, coefficients, x_star)
        exact = float(cotangent(x_star))

        print(f"\nСлучай {case['name']}")
        print("Исходные узлы и значения функции:")
        for i, (label, node, value) in enumerate(zip(labels, nodes, values)):
            print(f"  x{i} = {label} = {node:.{precision}g}; y{i} = {value:.{precision}g}")
        print(
            f"Многочлен Лагранжа L3(x) = "
            f"{lagrange_polynomial_text(nodes, values, labels)}"
        )
        print_divided_differences(table)
        print(
            f"Многочлен Ньютона N3(x) = "
            f"{newton_polynomial_text(coefficients, labels)}"
        )
        print(f"L3(x*) = {lagrange_result:.{precision}g}")
        print(f"N3(x*) = {newton_result:.{precision}g}")
        print(f"f(x*) = {exact:.{precision}g}")
        print(f"|f(x*) - L3(x*)| = {abs(exact - lagrange_result):.{precision}g}")
        print(f"|f(x*) - N3(x*)| = {abs(exact - newton_result):.{precision}g}")
        print(f"|L3(x*) - N3(x*)| = {abs(lagrange_result - newton_result):.{precision}g}")

        plot_path = BASE_DIR / case["plot_file"]
        create_plot(
            nodes, values, coefficients, x_star, pi_label(x_star_fraction),
            labels, case["name"], plot_path,
        )
        print(f"График: {plot_path.name}")


if __name__ == "__main__":
    main()
