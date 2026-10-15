"""Решение нелинейного уравнения методами итераций и Ньютона."""

import numpy as np

from utils import read_epsilon, setup_output, show


INPUT_FILE = "input.json"
MAX_ITERATIONS = 10000
LEFT_BOUND = 0.5
RIGHT_BOUND = 1.0
INITIAL_GUESS = 1.0
ZERO_TOLERANCE = 1e-12
PLOT_FILE = "nonlinear_equation.png"
ERROR_PLOT_FILE = "nonlinear_equation_errors.png"


def function(x):
    """Вычислить значение исходной функции."""
    return x ** 3 + x ** 2 - x - 0.5


def derivative(x):
    """Вычислить значение первой производной."""
    return 3 * x ** 2 + 2 * x - 1


def iteration_function(x):
    """Вычислить следующее приближение метода простой итерации."""
    return np.cbrt(-x ** 2 + x + 0.5)


def iteration_derivative(x):
    """Вычислить производную итерационной функции."""
    radicand = -x ** 2 + x + 0.5
    return (1 - 2 * x) / (3 * np.cbrt(radicand) ** 2)


def simple_iterations(initial, epsilon, max_iterations, contraction):
    """Найти корень методом простой итерации."""
    if not 0 <= contraction < 1:
        raise ValueError("на выбранном отрезке не выполнено условие сходимости")

    x = initial
    history = []
    for iteration in range(1, max_iterations + 1):
        next_x = iteration_function(x)
        error = contraction / (1 - contraction) * abs(next_x - x)
        history.append((iteration, next_x, error, abs(function(next_x))))
        if error <= epsilon:
            return next_x, iteration, history
        x = next_x
    raise ValueError("метод простой итерации не сошёлся")


def newton_method(initial, epsilon, max_iterations):
    """Найти корень методом Ньютона."""
    x = initial
    history = []
    for iteration in range(1, max_iterations + 1):
        derivative_value = derivative(x)
        if abs(derivative_value) <= ZERO_TOLERANCE:
            raise ValueError("производная метода Ньютона близка к нулю")

        next_x = x - function(x) / derivative_value
        error = abs(next_x - x)
        history.append((iteration, next_x, error, abs(function(next_x))))
        if error <= epsilon:
            return next_x, iteration, history
        x = next_x
    raise ValueError("метод Ньютона не сошёлся")


def compare_methods(simple_count, newton_count):
    """Сравнить количество итераций двух методов."""
    if simple_count == newton_count:
        return "Методы сошлись за одинаковое количество итераций."
    if simple_count < newton_count:
        return "Метод простой итерации потребовал меньше итераций."
    return "Метод Ньютона потребовал меньше итераций."


def create_plot(filename):
    """Построить график функции для выбора начального приближения."""
    import matplotlib.pyplot as plt

    x_values = np.linspace(0, 1.2, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, function(x_values), label="f(x) = x³ + x² - x - 0.5")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(INITIAL_GUESS, color="gray", linestyle="--", label="x₀ = 1")
    plt.scatter(
        [LEFT_BOUND, RIGHT_BOUND],
        [function(LEFT_BOUND), function(RIGHT_BOUND)],
        color="red",
        zorder=3,
        label="границы отрезка",
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def create_error_plot(simple_history, newton_history, epsilon, filename):
    """Построить график зависимости погрешности от номера итерации."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for history, label in (
        (simple_history, "Простая итерация"),
        (newton_history, "Метод Ньютона"),
    ):
        iterations = [item[0] for item in history]
        errors = [item[2] for item in history]
        plt.semilogy(iterations, errors, marker="o", label=label)

    plt.axhline(
        epsilon,
        color="red",
        linestyle="--",
        label=f"ε = {epsilon:g}",
    )
    plt.xlabel("Номер итерации k")
    plt.ylabel("Оценка погрешности")
    iterations = {
        item[0]
        for history in (simple_history, newton_history)
        for item in history
    }
    plt.xticks(sorted(iterations))
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    try:
        epsilon = read_epsilon(INPUT_FILE)
        contraction = max(
            abs(iteration_derivative(LEFT_BOUND)),
            abs(iteration_derivative(RIGHT_BOUND)),
        )
        simple_root, simple_count, simple_history = simple_iterations(
            INITIAL_GUESS,
            epsilon,
            MAX_ITERATIONS,
            contraction,
        )
        newton_root, newton_count, newton_history = newton_method(
            INITIAL_GUESS,
            epsilon,
            MAX_ITERATIONS,
        )
        create_plot(PLOT_FILE)
        create_error_plot(
            simple_history,
            newton_history,
            epsilon,
            ERROR_PLOT_FILE,
        )

        setup_output(precision=10)
        print("Уравнение: x³ + x² - x - 0.5 = 0")
        print(f"Заданная точность ε: {epsilon:g}")
        print(f"Отрезок локализации корня: [{LEFT_BOUND:g}, {RIGHT_BOUND:g}]")
        print(f"Начальное приближение x₀: {INITIAL_GUESS:g}")
        print(f"Коэффициент сжатия q: {contraction:.6g}")
        show("Корень методом простой итерации", simple_root)
        show("Невязка метода простой итерации", abs(function(simple_root)))
        print(f"Количество итераций: {simple_count}")
        show("Корень методом Ньютона", newton_root)
        show("Невязка метода Ньютона", abs(function(newton_root)))
        print(f"Количество итераций: {newton_count}")
        show("Сравнение скорости сходимости", compare_methods(simple_count, newton_count))
        print(f"\nГрафик функции сохранён в файл {PLOT_FILE}")
        print(f"График погрешности сохранён в файл {ERROR_PLOT_FILE}")
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
