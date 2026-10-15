"""Решение нелинейной системы методами итераций и Ньютона."""

import numpy as np

from utils import read_epsilon, setup_output, show


A_PARAMETER = 1.0
INPUT_FILE = "input.json"
MAX_ITERATIONS = 10000
INITIAL_GUESS = np.array([1.2, 1.35])
X1_BOUNDS = (1.15, 1.3)
X2_BOUNDS = (1.3, 1.4)
ZERO_TOLERANCE = 1e-12
PLOT_FILE = "nonlinear_system.png"
ERROR_PLOT_FILE = "nonlinear_system_errors.png"


def system(values):
    """Вычислить значения функций системы."""
    x1, x2 = values
    if x1 <= -1:
        raise ValueError("аргумент логарифма должен быть положительным")
    return np.array([
        x1 - np.cos(x2) - 1,
        x2 - np.log10(x1 + 1) - A_PARAMETER,
    ])


def iteration_function(values):
    """Вычислить следующее итерационное приближение."""
    x1, x2 = values
    if x1 <= -1:
        raise ValueError("аргумент логарифма должен быть положительным")
    return np.array([
        1 + np.cos(x2),
        A_PARAMETER + np.log10(x1 + 1),
    ])


def jacobian(values):
    """Построить матрицу Якоби исходной системы."""
    x1, x2 = values
    if x1 <= -1:
        raise ValueError("аргумент логарифма должен быть положительным")
    return np.array([
        [1, np.sin(x2)],
        [-1 / ((x1 + 1) * np.log(10)), 1],
    ])


def contraction_coefficient():
    """Оценить норму производной итерационной функции."""
    log_derivative = 1 / ((X1_BOUNDS[0] + 1) * np.log(10))
    cosine_derivative = max(abs(np.sin(value)) for value in X2_BOUNDS)
    return max(log_derivative, cosine_derivative)


def simple_iterations(initial, epsilon, max_iterations, contraction):
    """Решить систему методом простой итерации."""
    if not 0 <= contraction < 1:
        raise ValueError(
            "на выбранной области не выполнено условие сходимости"
        )

    values = initial.copy()
    history = []
    for iteration in range(1, max_iterations + 1):
        next_values = iteration_function(values)
        difference = np.max(np.abs(next_values - values))
        error = contraction / (1 - contraction) * difference
        residual = np.max(np.abs(system(next_values)))
        history.append((iteration, next_values.copy(), error, residual))
        if error <= epsilon:
            return next_values, iteration, history
        values = next_values
    raise ValueError("метод простой итерации не сошёлся")


def solve_newton_system(matrix, right):
    """Найти поправку Ньютона из линейной системы 2 × 2."""
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(determinant) <= ZERO_TOLERANCE:
        raise ValueError("матрица Якоби вырождена")

    delta1 = (right[0] * matrix[1, 1] - matrix[0, 1] * right[1]) / determinant
    delta2 = (matrix[0, 0] * right[1] - right[0] * matrix[1, 0]) / determinant
    return np.array([delta1, delta2])


def newton_method(initial, epsilon, max_iterations):
    """Решить систему методом Ньютона."""
    values = initial.copy()
    history = []
    for iteration in range(1, max_iterations + 1):
        correction = solve_newton_system(jacobian(values), -system(values))
        next_values = values + correction
        error = np.max(np.abs(next_values - values))
        residual = np.max(np.abs(system(next_values)))
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


def create_plot(solution, filename):
    """Построить графики уравнений системы."""
    import matplotlib.pyplot as plt

    x2_values = np.linspace(1.1, 1.6, 400)
    first_x1 = 1 + np.cos(x2_values)
    x1_values = np.linspace(0.9, 1.5, 400)
    second_x2 = A_PARAMETER + np.log10(x1_values + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(first_x1, x2_values, label="x₁ = 1 + cos(x₂)")
    plt.plot(x1_values, second_x2, label="x₂ = 1 + lg(x₁ + 1)")
    plt.scatter(
        *INITIAL_GUESS,
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
        contraction = contraction_coefficient()
        simple_solution, simple_count, simple_history = simple_iterations(
            INITIAL_GUESS,
            epsilon,
            MAX_ITERATIONS,
            contraction,
        )
        newton_solution, newton_count, newton_history = newton_method(
            INITIAL_GUESS,
            epsilon,
            MAX_ITERATIONS,
        )
        create_plot(newton_solution, PLOT_FILE)
        create_error_plot(
            simple_history,
            newton_history,
            epsilon,
            ERROR_PLOT_FILE,
        )

        setup_output(precision=10)
        print("Система:")
        print("x₁ - cos(x₂) = 1")
        print(f"x₂ - lg(x₁ + 1) = {A_PARAMETER:g}")
        print(f"Заданная точность ε: {epsilon:g}")
        show("Начальное приближение", INITIAL_GUESS)
        print(f"Коэффициент сжатия q: {contraction:.6g}")
        show("Решение простыми итерациями", simple_solution)
        show("Невязка простых итераций", system(simple_solution))
        print(f"Количество итераций: {simple_count}")
        show("Решение методом Ньютона", newton_solution)
        show("Невязка метода Ньютона", system(newton_solution))
        print(f"Количество итераций: {newton_count}")
        comparison = compare_methods(simple_count, newton_count)
        show("Сравнение скорости сходимости", comparison)
        print(f"\nГрафик системы сохранён в файл {PLOT_FILE}")
        print(f"График погрешности сохранён в файл {ERROR_PLOT_FILE}")
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
