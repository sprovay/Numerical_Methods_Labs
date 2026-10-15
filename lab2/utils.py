import json

import numpy as np


def read_json(filename):
    """Прочитать конфигурацию из JSON-файла."""
    with open(filename, encoding="utf-8") as file:
        return json.load(file)


def setup_output(precision=6):
    np.set_printoptions(precision=precision, suppress=True)


def show(title, value):
    print(f"\n{title}:")
    print(value)
