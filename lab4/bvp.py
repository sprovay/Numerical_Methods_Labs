"""Лабораторная 4.2, вариант 4: краевая задача для ОДУ второго порядка."""
# Было - y(2) = -1 + 3ln(2), y = -1 + ...; Стало - y(2) = -1 + 3ln(3), y = -x + ....

import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ode_utils import make_grid, rk4_step, runge_romberg


sys.path.insert(0, "../lab1")
from sweep_method import sweep_method


LEFT_VALUE = 1 + 4 * math.log(2)
RIGHT_VALUE = -1 + 3 * math.log(3)


def exact_solution(x):
    """Точное решение краевой задачи из исправленного условия."""
    x = np.asarray(x, dtype=float)
    return -x + 2 / x + 2 * (x + 1) / x * np.log(x + 1)


def right_side(x, state):
    y, derivative = state
    return np.array([derivative, 2 * y / (x * x * (x + 1))])


def integrate_with_slope(grid, slope):
    states = np.empty((len(grid), 2))
    states[0] = [LEFT_VALUE, slope]
    for i in range(len(grid) - 1):
        states[i + 1] = rk4_step(
            right_side, grid[i], states[i], grid[i + 1] - grid[i]
        )
    return states[:, 0]


def shooting(grid):
    """Подобрать y'(1) методом секущих; интегрировать IVP методом RK4."""
    first_slope, second_slope = 0.0, 1.0
    first_values = integrate_with_slope(grid, first_slope) # две задачи Коши
    second_values = integrate_with_slope(grid, second_slope)
    first_residual = first_values[-1] - RIGHT_VALUE # невязка
    second_residual = second_values[-1] - RIGHT_VALUE
    for _ in range(20):
        denominator = second_residual - first_residual
        if abs(denominator) < 1e-14:
            raise ValueError("метод стрельбы: невозможно обновить начальный наклон")
        next_slope = second_slope - second_residual * ( # строим прямую через 2 точки, пересечение с s дает новое значение slope
            second_slope - first_slope
        ) / denominator
        values = integrate_with_slope(grid, next_slope)
        residual = values[-1] - RIGHT_VALUE
        if abs(residual) < 1e-12:
            return values, next_slope, residual
        first_slope, first_residual = second_slope, second_residual
        second_slope, second_residual = next_slope, residual
    raise ValueError("метод стрельбы не сошёлся")


def finite_difference(grid):
    """Центральная разность второго порядка и метод прогонки из lab1."""
    h = grid[1] - grid[0]
    interior = grid[1:-1]
    if len(interior) == 0:
        raise ValueError("для конечно-разностного метода нужен внутренний узел")
    coefficient = interior * interior * (interior + 1) / (h * h)
    lower = coefficient[1:]
    diagonal = -2 * coefficient - 2
    upper = coefficient[:-1]
    right = np.zeros(len(interior))
    right[0] -= coefficient[0] * LEFT_VALUE
    right[-1] -= coefficient[-1] * RIGHT_VALUE
    values = np.empty(len(grid))
    values[0], values[-1] = LEFT_VALUE, RIGHT_VALUE
    values[1:-1], _, _ = sweep_method(lower, diagonal, upper, right)
    return values


METHODS = (
    ("Метод стрельбы", 4),
    ("Конечно-разностный метод", 2),
)


def read_config():
    with open("bvp_input.json", encoding="utf-8") as file:
        config = json.load(file)
    left, right = map(float, config["interval"])
    h = float(config["h"])
    ratio = int(config["refinement_factor"])
    precision = int(config["output_precision"])
    if left != 1 or right != 2:
        raise ValueError("для варианта 4 отрезок должен быть [1, 2]")
    if ratio < 2 or not 1 <= precision <= 16:
        raise ValueError("нужны refinement_factor >= 2 и точность вывода от 1 до 16")
    coarse = make_grid(left, right, h)
    fine = make_grid(left, right, h / ratio)
    if len(coarse) < 3:
        raise ValueError("на крупной сетке нужен хотя бы один внутренний узел")
    return config, coarse, fine, ratio, precision


def create_plots(config, coarse, fine, results):
    dense = np.linspace(coarse[0], coarse[-1], 600)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ("tab:blue", "tab:orange")
    markers = ("o", "x")
    ax.plot(dense, exact_solution(dense), color="black", linewidth=2,
            label="Точное решение")
    for (name, _), color, marker in zip(METHODS, colors, markers):
        data = results[name]
        ax.plot(fine, data["fine"], color=color, linewidth=1.6,
                label=f"{name}, h={fine[1]-fine[0]:g}")
        ax.scatter(coarse, data["coarse"], color=color, marker=marker, s=40,
                   label=f"{name}, h={coarse[1]-coarse[0]:g}")
    ax.set(xlabel="x", ylabel="y", title="Краевая задача, вариант 4")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config["solution_plot"], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(9, 7), sharex=True)
    for ax, (name, _), color in zip(axes, METHODS, colors):
        ax.plot(fine, np.abs(results[name]["fine"] - exact_solution(fine)),
                color=color, marker=".", label=name)
        ax.set_ylabel("|ошибка|")
        ax.grid(True, alpha=0.35)
        ax.legend()
    axes[0].set_title("Абсолютные ошибки на мелкой сетке, задача 4.2")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(config["error_plot"], dpi=160)
    plt.close(fig)


def main():
    config, coarse, fine, ratio, precision = read_config()
    exact_coarse = exact_solution(coarse)
    exact_fine = exact_solution(fine)
    results = {}

    print("Задача 4.2, вариант 4: x²(x+1)y'' - 2y = 0")
    print(f"y(1) = {LEFT_VALUE:.{precision}g}, y(2) = {RIGHT_VALUE:.{precision}g}")
    print("Точное решение: y(x) = -x + 2/x + 2(x+1)ln(x+1)/x\n")

    coarse_shoot, coarse_slope, coarse_residual = shooting(coarse)
    fine_shoot, fine_slope, fine_residual = shooting(fine)
    print(f"Стрельба: y'(1) при h={coarse[1]-coarse[0]:g}: "
          f"{coarse_slope:.{precision}g}; невязка y(2): {abs(coarse_residual):.{precision}g}")
    print(f"Стрельба: y'(1) при h={fine[1]-fine[0]:g}: "
          f"{fine_slope:.{precision}g}; невязка y(2): {abs(fine_residual):.{precision}g}\n")
    values_by_method = {
        "Метод стрельбы": (coarse_shoot, fine_shoot),
        "Конечно-разностный метод": (finite_difference(coarse), finite_difference(fine)),
    }

    for name, order in METHODS:
        coarse_values, fine_values = values_by_method[name]
        fine_shared = fine_values[::ratio]
        refined, rr_error = runge_romberg(coarse_values, fine_shared, order, ratio)
        results[name] = {"coarse": coarse_values, "fine": fine_values}
        print(f"{name} (p={order})")
        print(f"{'x':>7} {'y(h)':>15} {'|err(h)|':>14} {'y(h/r)':>15} "
              f"{'|err(h/r)|':>14} {'RR оценка':>14}")
        for i, x in enumerate(coarse):
            print(f"{x:7.3f} {coarse_values[i]:15.{precision}g} "
                  f"{abs(coarse_values[i]-exact_coarse[i]):14.{precision}g} "
                  f"{fine_shared[i]:15.{precision}g} "
                  f"{abs(fine_shared[i]-exact_coarse[i]):14.{precision}g} "
                  f"{rr_error[i]:14.{precision}g}")
        print(f"Макс. |ошибка| при h: {np.max(np.abs(coarse_values-exact_coarse)):.{precision}g}")
        print(f"Макс. |ошибка| при h/r: {np.max(np.abs(fine_values-exact_fine)):.{precision}g}")
        print(f"Макс. оценка Рунге–Ромберга для h/r: {np.max(rr_error):.{precision}g}")
        print(f"Макс. |ошибка уточнения|: {np.max(np.abs(refined-exact_coarse)):.{precision}g}\n")

    create_plots(config, coarse, fine, results)
    print(f"Графики: {config['solution_plot']}, {config['error_plot']}")


if __name__ == "__main__":
    main()
