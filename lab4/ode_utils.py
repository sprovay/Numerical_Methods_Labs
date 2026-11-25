"""Общие операции для заданий 4.1 и 4.2."""

import numpy as np


def make_grid(left, right, step):
    """Построить равномерную сетку, если шаг точно делит отрезок."""
    if not np.isfinite(step) or step <= 0 or left >= right:
        raise ValueError("нужны left < right и положительный конечный шаг")
    intervals = round((right - left) / step)
    if intervals < 1 or not np.isclose(left + intervals * step, right, rtol=0, atol=1e-10):
        raise ValueError("шаг должен делить отрезок на целое число частей")
    return np.linspace(left, right, intervals + 1)


def rk4_step(function, x, state, step):
    """Один шаг классического метода Рунге — Кутты 4-го порядка."""
    k1 = function(x, state)
    k2 = function(x + step / 2, state + step * k1 / 2)
    k3 = function(x + step / 2, state + step * k2 / 2)
    k4 = function(x + step, state + step * k3)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def runge_romberg(coarse, fine_at_coarse_nodes, order, ratio=2):
    """Вернуть уточнение и оценку ошибки решения на мелкой сетке."""
    correction = (fine_at_coarse_nodes - coarse) / (ratio ** order - 1)
    return fine_at_coarse_nodes + correction, np.abs(correction)
