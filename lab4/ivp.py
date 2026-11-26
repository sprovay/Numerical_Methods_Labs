"""Лабораторная 4.1, вариант 4: задача Коши для ОДУ второго порядка."""
# Было - y'(1) = 2e^(1/2) - 1; Стало - y'(1) = -1.

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ode_utils import make_grid, rk4_step, runge_romberg


INITIAL_Y = 1 + math.exp(0.5)
INITIAL_DY = -1


def exact_solution(x):
    """Точное решение для y(1)=1+exp(1/2), y'(1)=-1."""
    x = np.asarray(x, dtype=float)
    return (1 + np.exp(x * x / 2)) / x


def right_side(x, state):
    """y'' = ((x^3-x)y' + (x^2+1)y) / x^2; state = [y, y']."""
    y, derivative = state
    return np.array([
        derivative,
        (x * (x * x - 1) * derivative + (x * x + 1) * y) / (x * x),
    ])


def euler(grid, initial):
    states = np.empty((len(grid), 2))
    states[0] = initial
    for i in range(len(grid) - 1):
        h = grid[i + 1] - grid[i]
        states[i + 1] = states[i] + h * right_side(grid[i], states[i])
    return states


def runge_kutta(grid, initial):
    states = np.empty((len(grid), 2))
    states[0] = initial
    for i in range(len(grid) - 1):
        states[i + 1] = rk4_step(
            right_side, grid[i], states[i], grid[i + 1] - grid[i]
        )
    return states


def adams_bashforth_4(grid, initial):
    """Явная формула Адамса 4-го порядка; первые 3 шага — RK4."""
    if len(grid) < 4:
        raise ValueError("для метода Адамса нужны хотя бы три шага")
    states = np.empty((len(grid), 2))
    states[0] = initial
    h = grid[1] - grid[0]
    for i in range(3):
        states[i + 1] = rk4_step(right_side, grid[i], states[i], h)
    for i in range(3, len(grid) - 1):
        f0 = right_side(grid[i], states[i])
        f1 = right_side(grid[i - 1], states[i - 1])
        f2 = right_side(grid[i - 2], states[i - 2])
        f3 = right_side(grid[i - 3], states[i - 3])
        states[i + 1] = states[i] + h * (
            55 * f0 - 59 * f1 + 37 * f2 - 9 * f3
        ) / 24
    return states


METHODS = (
    ("Эйлер", 1, euler),
    ("Рунге–Кутта 4", 4, runge_kutta),
    ("Адамс 4", 4, adams_bashforth_4),
)


def read_config():
    with open("ivp_input.json", encoding="utf-8") as file:
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
    if len(coarse) < 4:
        raise ValueError("для метода Адамса нужны хотя бы три шага на крупной сетке")
    return config, coarse, fine, ratio, precision


def create_plots(config, coarse, fine, results):
    dense = np.linspace(coarse[0], coarse[-1], 600)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ("tab:blue", "tab:orange", "tab:green")
    ax.plot(dense, exact_solution(dense), color="black", linewidth=2,
            label="Точное решение")
    for (name, _, _), color in zip(METHODS, colors):
        data = results[name]
        ax.plot(fine, data["fine"], color=color, linewidth=1.7,
                label=f"{name}, h={fine[1] - fine[0]:g}")
        ax.scatter(coarse, data["coarse"], color=color, s=22,
                   label=f"{name}, h={coarse[1] - coarse[0]:g}")
    ax.set(xlabel="x", ylabel="y", title="Задача Коши, вариант 4")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config["solution_plot"], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(9, 8), sharex=True)
    for ax, (name, _, _), color in zip(axes, METHODS, colors):
        ax.plot(fine, np.abs(results[name]["fine"] - exact_solution(fine)),
                color=color, marker=".", label=name)
        ax.set_ylabel("|ошибка|")
        ax.grid(True, alpha=0.35)
        ax.legend()
    axes[0].set_title("Абсолютные ошибки на мелкой сетке, задача 4.1")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(config["error_plot"], dpi=160)
    plt.close(fig)


def main():
    config, coarse, fine, ratio, precision = read_config()
    initial = np.array([INITIAL_Y, INITIAL_DY], dtype=float)
    exact_coarse = exact_solution(coarse)
    exact_fine = exact_solution(fine)
    results = {}

    print("Задача 4.1, вариант 4: x²y'' - x(x²-1)y' - (x²+1)y = 0")
    print(f"y(1) = {INITIAL_Y:.{precision}g}, y'(1) = {INITIAL_DY:.{precision}g}")
    print("Точное решение: y(x) = [1 + exp(x²/2)] / x\n")

    for name, order, method in METHODS:
        coarse_values = method(coarse, initial)[:, 0]
        fine_values = method(fine, initial)[:, 0]
        fine_shared = fine_values[::ratio]
        refined, rr_error = runge_romberg(coarse_values, fine_shared, order, ratio)
        results[name] = {"coarse": coarse_values, "fine": fine_values}

        print(f"{name} (p={order}, h={coarse[1]-coarse[0]:g} и "
              f"h={fine[1]-fine[0]:g})")
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

    create_plots(config, coarse, fine, results)
    print(f"Графики: {config['solution_plot']}, {config['error_plot']}")


if __name__ == "__main__":
    main()
