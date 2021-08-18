# -*- coding: utf-8 -*-
"""
This script is to compare gradient descent and stochastic gradient decent in terms
of time and cost optimization, finally save a graph where axis x represents the time
in minutes and axis y represents the value in cost function.
"""
import time
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import glove.cost_function
import glove.gradient
import glove.fit
import utils.util


# import inputs
base = "C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/files/spanish/300d_5w/"
file_name = "co_occurrence"

file_path = os.path.join(base, file_name + ".json")
with open(file_path, "r") as f:
    co_occurrence = json.load(f)

vocabulary = utils.util.get_vocabulary(co_occurrence)

DIMENSION = 300
theta_SGD = utils.util.gen_theta_dict(vocabulary, DIMENSION)
theta_GD = utils.util.gen_theta_dict(vocabulary, DIMENSION)

learning_rate = 0.08
factor = max(co_occurrence.values()) * 0.2

print(f"optimizing theta ... with a learning rate = {learning_rate}")
hist_cost_SGD = [
    glove.cost_function.cost_glove_dict(theta_SGD, co_occurrence, factor=factor)
]
minutes_SGD = [0]
acum = 0
for i in range(3):
    print(f"Iteration n°: {i}")
    inicio = time.time()
    gradient_SGD = glove.gradient.stochastic_gradient_descent(
        vocabulary, theta_SGD, co_occurrence, factor=factor
    )
    acum += (time.time() - inicio) / 60
    print("Maximo gradiente", np.max(gradient_SGD))

    theta_SGD = theta_SGD - learning_rate * gradient_SGD

    if i % 3 == 0:
        cost_model = glove.cost_function.cost_glove_dict(
            theta_SGD, co_occurrence, factor=factor
        )
        hist_cost_SGD.append(cost_model)
        minutes_SGD.append(acum)
        if hist_cost_SGD[-1] > hist_cost_SGD[-2]:
            print("Stop - increasing cost")
            break


learning_rate = 0.0008
print(f"optimizing theta ... with a learning rate = {learning_rate}")
hist_cost_GD = [glove.cost_function.cost_glove_dict(theta_GD, co_occurrence, factor)]
minutes_GD = [0]
acum = 0
for i in range(3):
    print(f"Iteration n°: {i}")
    inicio = time.time()
    gradient_GD = glove.gradient.gradient_descent_dict(
        vocabulary, theta_GD, co_occurrence, factor
    )
    acum += (time.time() - inicio) / 60
    print(acum)
    theta_GD = glove.fit.update_theta(theta_GD, gradient_GD, learning_rate)

    if i % 3 == 0:
        cost_model = glove.cost_function.cost_glove_dict(
            theta_GD, co_occurrence, factor
        )
        hist_cost_GD.append(cost_model)
        minutes_GD.append(acum)
        if hist_cost_GD[-1] > hist_cost_GD[-2]:
            print("Stop - increasing cost")
            break

plt.plot(range(len(hist_cost_SGD)), hist_cost_SGD, label="SGD")
plt.plot(range(len(hist_cost_GD)), hist_cost_GD, label="GD")
plt.title("Learning - SGD vs GD")
plt.xlabel("Minutes")
plt.ylabel("Cost")
plt.legend()
plt.savefig("SGD_GD.png")
plt.show()
