"""Решение нелинейного уравнения методами итераций и Ньютона."""

import numpy as np

from utils import read_json, setup_output, show


INPUT_FILE = "nonlinear_equation_input.json"


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


def newton_method(initial, epsilon, max_iterations, zero_tolerance):
    """Найти корень методом Ньютона."""
    x = initial
    history = []
    for iteration in range(1, max_iterations + 1):
        derivative_value = derivative(x)
        if abs(derivative_value) <= zero_tolerance:
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


def create_plot(initial, bounds, plot_config):
    """Построить график функции для выбора начального приближения."""
    import matplotlib.pyplot as plt

    left_bound, right_bound = bounds
    x_values = np.linspace(*plot_config["x_range"], plot_config["points"])
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, function(x_values), label="f(x) = x³ + x² - x - 0.5")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(initial, color="gray", linestyle="--", label=f"x₀ = {initial:g}")
    plt.axvline(
        left_bound,
        color="red",
        linestyle="--",
        label="границы отрезка",
    )
    plt.axvline(right_bound, color="red", linestyle="--")
    plt.scatter(
        [left_bound, right_bound],
        [function(left_bound), function(right_bound)],
        color="red",
        zorder=3,
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_config["function_file"], dpi=plot_config["dpi"])
    plt.close()


def create_error_plot(
    simple_history,
    newton_history,
    reference_root,
    plot_config,
):
    """Построить график ошибки относительно эталонного корня."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for history, label in (
        (simple_history, "Простая итерация"),
        (newton_history, "Метод Ньютона"),
    ):
        iterations = [item[0] for item in history]
        errors = [abs(item[1] - reference_root) for item in history]
        plt.semilogy(iterations, errors, marker="o", label=label)

    plt.xlabel("Номер итерации k")
    plt.ylabel("Погрешность приближения")
    iterations = {
        item[0]
        for history in (simple_history, newton_history)
        for item in history
    }
    plt.xticks(sorted(iterations))
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_config["error_file"], dpi=plot_config["dpi"])
    plt.close()


def main():
    try:
        config = read_json(INPUT_FILE)
        epsilon = float(config["epsilon"])
        reference_epsilon = np.longdouble(epsilon) ** 2
        max_iterations = int(config["max_iterations"])
        left_bound, right_bound = map(float, config["interval"])
        initial_guess = float(config["initial_guess"])
        zero_tolerance = float(config["zero_tolerance"])
        plot_config = config["plot"]

        if epsilon <= 0 or max_iterations <= 0 or zero_tolerance <= 0:
            raise ValueError("точности и число итераций должны быть положительными")
        if left_bound >= right_bound:
            raise ValueError("левая граница должна быть меньше правой")

        contraction = max(
            abs(iteration_derivative(left_bound)),
            abs(iteration_derivative(right_bound)),
        )
        simple_root, simple_count, simple_history = simple_iterations(
            initial_guess,
            epsilon,
            max_iterations,
            contraction,
        )
        newton_root, newton_count, newton_history = newton_method(
            initial_guess,
            epsilon,
            max_iterations,
            zero_tolerance,
        )
        reference_root, _, _ = newton_method(
            np.longdouble(initial_guess),
            reference_epsilon,
            max_iterations,
            np.longdouble(zero_tolerance),
        )
        create_plot(initial_guess, (left_bound, right_bound), plot_config)
        create_error_plot(
            simple_history,
            newton_history,
            reference_root,
            plot_config,
        )

        setup_output(precision=int(config["output_precision"]))
        print("Уравнение: x³ + x² - x - 0.5 = 0")
        print(f"Заданная точность ε: {epsilon:g}")
        print(f"Отрезок локализации корня: [{left_bound:g}, {right_bound:g}]")
        print(f"Начальное приближение x₀: {initial_guess:g}")
        print(f"Коэффициент сжатия q: {contraction:.6g}")
        show("Корень методом простой итерации", simple_root)
        show("Невязка метода простой итерации", abs(function(simple_root)))
        print(f"Количество итераций: {simple_count}")
        show("Корень методом Ньютона", newton_root)
        show("Невязка метода Ньютона", abs(function(newton_root)))
        print(f"Количество итераций: {newton_count}")
        show("Сравнение скорости сходимости", compare_methods(simple_count, newton_count))
        print(f"\nГрафик функции сохранён в файл {plot_config['function_file']}")
        print(f"График погрешности сохранён в файл {plot_config['error_file']}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
