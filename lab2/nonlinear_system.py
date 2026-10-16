"""Решение нелинейной системы методами итераций и Ньютона."""

import numpy as np

from utils import read_json, setup_output, show


INPUT_FILE = "nonlinear_system_input.json"


def system(values, parameter):
    """Вычислить значения функций системы."""
    x1, x2 = values
    if x1 <= -1:
        raise ValueError("аргумент логарифма должен быть положительным")
    return np.array([
        x1 - np.cos(x2) - 1,
        x2 - np.log10(x1 + 1) - parameter,
    ])


def iteration_function(values, parameter):
    """Вычислить следующее итерационное приближение."""
    x1, x2 = values
    if x1 <= -1:
        raise ValueError("аргумент логарифма должен быть положительным")
    return np.array([
        1 + np.cos(x2),
        parameter + np.log10(x1 + 1),
    ])


def jacobian(values):
    """Построить матрицу Якоби исходной системы."""
    x1, x2 = values
    if x1 <= -1:
        raise ValueError("аргумент логарифма должен быть положительным")
    return np.array([
        [1, np.sin(x2)],
        [-1 / ((x1 + 1) * np.log(np.longdouble(10))), 1],
    ])


def contraction_coefficient(x1_bounds, x2_bounds):
    """Оценить норму производной итерационной функции."""
    log_derivative = 1 / ((x1_bounds[0] + 1) * np.log(10))
    cosine_derivative = max(abs(np.sin(value)) for value in x2_bounds)
    return max(log_derivative, cosine_derivative)


def simple_iterations(initial, epsilon, max_iterations, contraction, parameter):
    """Решить систему методом простой итерации."""
    if not 0 <= contraction < 1:
        raise ValueError(
            "на выбранной области не выполнено условие сходимости"
        )

    values = initial.copy()
    history = []
    for iteration in range(1, max_iterations + 1):
        next_values = iteration_function(values, parameter)
        difference = np.max(np.abs(next_values - values))
        error = contraction / (1 - contraction) * difference
        residual = np.max(np.abs(system(next_values, parameter)))
        history.append((iteration, next_values.copy(), error, residual))
        if error <= epsilon:
            return next_values, iteration, history
        values = next_values
    raise ValueError("метод простой итерации не сошёлся")


def solve_newton_system(matrix, right, zero_tolerance):
    """Найти поправку Ньютона из линейной системы 2 × 2."""
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(determinant) <= zero_tolerance:
        raise ValueError("матрица Якоби вырождена")

    delta1 = (right[0] * matrix[1, 1] - matrix[0, 1] * right[1]) / determinant
    delta2 = (matrix[0, 0] * right[1] - right[0] * matrix[1, 0]) / determinant
    return np.array([delta1, delta2])


def newton_method(initial, epsilon, max_iterations, parameter, zero_tolerance):
    """Решить систему методом Ньютона."""
    values = initial.copy()
    history = []
    for iteration in range(1, max_iterations + 1):
        correction = solve_newton_system(
            jacobian(values),
            -system(values, parameter),
            zero_tolerance,
        )
        next_values = values + correction
        error = np.max(np.abs(next_values - values))
        residual = np.max(np.abs(system(next_values, parameter)))
        history.append((iteration, next_values.copy(), error, residual))
        if error <= epsilon:
            return next_values, iteration, history
        values = next_values
    raise ValueError("метод Ньютона не сошёлся")


def compare_methods(simple_count, newton_count):
    """Сравнить количество итераций."""
    if simple_count == newton_count:
        return "Методы сошлись за одинаковое число итераций."
    if simple_count < newton_count:
        return "Метод простой итерации оказался быстрее."
    return "Метод Ньютона оказался быстрее."


def create_plot(
    solution,
    initial,
    parameter,
    x1_bounds,
    x2_bounds,
    plot_config,
):
    """Построить графики уравнений системы."""
    import matplotlib.pyplot as plt

    x2_values = np.linspace(*plot_config["x2_range"], plot_config["points"])
    first_x1 = 1 + np.cos(x2_values)
    x1_values = np.linspace(*plot_config["x1_range"], plot_config["points"])
    second_x2 = parameter + np.log10(x1_values + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(first_x1, x2_values, label="x₁ = 1 + cos(x₂)")
    plt.plot(x1_values, second_x2, label="x₂ = 1 + lg(x₁ + 1)")
    plt.axvline(
        x1_bounds[0],
        color="gray",
        linestyle="--",
        label="границы области D",
    )
    plt.axvline(x1_bounds[1], color="gray", linestyle="--")
    plt.axhline(x2_bounds[0], color="gray", linestyle="--")
    plt.axhline(x2_bounds[1], color="gray", linestyle="--")
    plt.scatter(
        *initial,
        color="orange",
        zorder=3,
        label="начальное приближение",
    )
    plt.scatter(*solution, color="red", zorder=3, label="решение")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_config["system_file"], dpi=plot_config["dpi"])
    plt.close()


def create_error_plot(
    simple_history,
    newton_history,
    reference_solution,
    plot_config,
):
    """Построить график ошибки относительно эталонного решения."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for history, label in (
        (simple_history, "Простая итерация"),
        (newton_history, "Метод Ньютона"),
    ):
        iterations = [item[0] for item in history]
        errors = [
            np.max(np.abs(item[1] - reference_solution))
            for item in history
        ]
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
        parameter = float(config["a_parameter"])
        initial_guess = np.array(config["initial_guess"], dtype=float)
        x1_bounds = tuple(map(float, config["x1_bounds"]))
        x2_bounds = tuple(map(float, config["x2_bounds"]))
        zero_tolerance = float(config["zero_tolerance"])
        plot_config = config["plot"]

        if epsilon <= 0 or max_iterations <= 0 or zero_tolerance <= 0:
            raise ValueError("точности и число итераций должны быть положительными")
        if initial_guess.shape != (2,):
            raise ValueError("начальное приближение должно содержать два числа")
        if x1_bounds[0] >= x1_bounds[1] or x2_bounds[0] >= x2_bounds[1]:
            raise ValueError("границы области заданы неверно")

        contraction = contraction_coefficient(x1_bounds, x2_bounds)
        simple_solution, simple_count, simple_history = simple_iterations(
            initial_guess,
            epsilon,
            max_iterations,
            contraction,
            parameter,
        )
        newton_solution, newton_count, newton_history = newton_method(
            initial_guess,
            epsilon,
            max_iterations,
            parameter,
            zero_tolerance,
        )
        reference_solution, _, _ = newton_method(
            initial_guess.astype(np.longdouble),
            reference_epsilon,
            max_iterations,
            np.longdouble(parameter),
            np.longdouble(zero_tolerance),
        )
        create_plot(
            newton_solution,
            initial_guess,
            parameter,
            x1_bounds,
            x2_bounds,
            plot_config,
        )
        create_error_plot(
            simple_history,
            newton_history,
            reference_solution,
            plot_config,
        )

        setup_output(precision=int(config["output_precision"]))
        print("Система:")
        print("x₁ - cos(x₂) = 1")
        print(f"x₂ - lg(x₁ + 1) = {parameter:g}")
        print(f"Заданная точность ε: {epsilon:g}")
        show("Начальное приближение", initial_guess)
        print(f"Коэффициент сжатия q: {contraction:.6g}")
        show("Решение простыми итерациями", simple_solution)
        show("Невязка простых итераций", system(simple_solution, parameter))
        print(f"Количество итераций: {simple_count}")
        show("Решение методом Ньютона", newton_solution)
        show("Невязка метода Ньютона", system(newton_solution, parameter))
        print(f"Количество итераций: {newton_count}")
        comparison = compare_methods(simple_count, newton_count)
        show("Сравнение скорости сходимости", comparison)
        print(f"\nГрафик системы сохранён в файл {plot_config['system_file']}")
        print(f"График погрешности сохранён в файл {plot_config['error_file']}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
