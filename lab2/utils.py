import json

import json

import numpy as np


def read_epsilon(filename):
    """Прочитать положительную конечную точность из JSON-файла."""
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)
    epsilon = float(data["epsilon"])
    return epsilon


def setup_output(precision=6):
    np.set_printoptions(precision=precision, suppress=True)


def show(title, value):
    print(f"\n{title}:")
    print(value)
