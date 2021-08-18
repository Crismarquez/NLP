# -*- coding: utf-8 -*-
"""
This script run the glove model using previos inputs created by get_inputs_for_glove.py
The main objective of the script is update theta and save this update to continue
using in the future, but using stocastic gradient descent (SGD) to optimize theta,
also save a json file with vector representation for context words.
In order to check the convergence of cost model, each three iteration the cost
will be calculated to then show in a graph.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import glove.cost_function
import glove.gradient


# import inputs
base = "files"
files_name = ["vocabulary", "co_occurrence", "theta"]
files = []
for file_name in files_name:
    file_path = os.path.join(base, file_name + ".json")
    with open(file_path, "r") as f:
        files.append(json.load(f)[file_name])

vocabulary = files[0]
co_occurrence = files[1]
theta = np.array(files[2])

print("Size of filtered vocabulary: ", "{:,.0f}".format(len(vocabulary)))
print("Size of theta: ", "{:,.0f}".format(len(theta)))

dimension = len(theta) // 2 // len(vocabulary)

learning_rate = 0.0005
print(f"optimizing theta ... with a learning rate = {learning_rate}")
hist_cost = [glove.cost_function.cost_glove_dict(vocabulary, theta, co_occurrence)]

file_path = os.path.join(base, "theta.json")
for i in range(20):
    print(f"Iteration n°: {i}")
    gradient = glove.gradient.stochastic_gradient_descent(
        vocabulary, theta, co_occurrence
    )

    max_grad = gradient.max()
    print("max value in gradient: ", max_grad)
    print("max change in theta: ", max_grad * learning_rate)

    theta = theta - learning_rate * gradient

    # # update file

    # with open(file_path, "w") as fp:
    #     json.dump({"theta": list(theta)}, fp)

    if i % 3 == 0:
        cost_model = glove.cost_function.cost_glove_dict(
            vocabulary, theta, co_occurrence
        )
        hist_cost.append(cost_model)

plt.plot(range(len(hist_cost)), hist_cost)
plt.title("Learning")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
