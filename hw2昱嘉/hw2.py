from math import copysign
from typing import List

import numpy as np


def generate_dataset(n: int, noise: float) -> List[List]:
    data = [[i, copysign(1, i)] for i in np.random.uniform(-1, 1, n)]
    if noise != 0:
        flip = np.random.binomial(1, [0.1] * len(data))
        for i in range(len(data)):
            if flip[i] == 1:
                data[i][1] *= -1

    return data


def generate_theta_and_s(data: List[List]) -> List[List]:
    result = [[-1, 1], [-1, -1]]
    for i in range(len(data) - 1):
        theta = (data[i][0] + data[i + 1][0]) / 2
        result.append([theta, 1])
        result.append([theta, -1])

    return result


def decision_stump(sampleSize: int, noise: float) -> float:
    trainData = sorted(generate_dataset(sampleSize, noise), key=lambda x: x[0])
    parameters = generate_theta_and_s(trainData)

    result = []
    for theta, s in parameters:
        errors = 0
        for x, y in trainData:
            if s * copysign(1, x - theta) != y:
                errors += 1
        result.append([theta, s, errors / sampleSize])

    bestTheta, _, ein = sorted(
        result, key=lambda x: (x[2], x[0] + x[1]))[0]

    return evaluate_eout(ein, bestTheta, noise) - ein


def evaluate_eout(ein: float, theta: float, noise: float) -> float:
    eout = abs(0.5 * theta)
    return eout * (1 - 2 * noise) + noise


def driver(sampleSize: int, noise: float) -> float:
    result = [decision_stump(sampleSize, noise) for _ in range(10000)]

    return sum(result) / len(result)


if __name__ == "__main__":
    print(
        f"Sample size: 2, noise: 0, Eout-Ein: {driver(sampleSize=2, noise=0)}")
    print(
        f"Sample size: 20, noise: 0, Eout-Ein: {driver(sampleSize=20, noise=0)}")
    print(
        f"Sample size: 2, noise: 0.1, Eout-Ein: {driver(sampleSize=2, noise=0.1)}")
    print(
        f"Sample size: 20, noise: 0.1, Eout-Ein: {driver(sampleSize=20, noise=0.1)}")
    print(
        f"Sample size: 200, noise: 0.1, Eout-Ein: {driver(sampleSize=200, noise=0.1)}")
