#!/usr/bin/env python
# coding: utf-8

# ## Question1 

# In[22]:


import numpy as np

# Parameters
n = 16  # Number of items
W = 25  # Capacity of knapsack
R = 50  # Penalty parameter
N = 20  # Population size
m = 6  # Number of children per iteration
max_iteration = 30  # Maximum number of iterations
p_c = 1  # Crossover probability
p_1 = 10**-3  # Mutation probability

# Values and weights of items
items = [
    (6, 3), (8, 5), (3, 4), (4, 7), (5, 4), (9, 10),
    (11, 3), (12, 6), (6, 8), (8, 14), (13, 4), (15, 9),
    (16, 10), (13, 11), (9, 17), (25, 12)
]
v = np.array([item[0] for item in items])
w = np.array([item[1] for item in items])

# Penalty function
def obj(x):
    total_value = np.dot(v, x)
    total_weight = np.dot(w, x)
    phi = max(0, total_weight - W)
    return total_value - R * phi

# Initialize population
pop = np.zeros((N, n + 1))  # Last column to hold function values
for i in range(N):
    x = np.random.randint(0, 2, size=n)
    pop[i, :n] = x
    pop[i, n] = obj(x)

# Sort population by objective function value (ascending order)
pop = pop[pop[:, n].argsort()]

# Main GA loop
for iteration in range(max_iteration):
    child = np.zeros((m, n + 1))
    
    # Generate offspring
    for k in range(0, m, 2):
        # Select two parents using tournament selection
        p = np.zeros(2, dtype=int)
        for j in range(2):
            n1, n2 = 0, 0
            while n1 == n2:
                n1 = np.random.randint(0, N)
                n2 = np.random.randint(0, N)
            p[j] = n1 if pop[n1, n] > pop[n2, n] else n2

        # Crossover
        x = pop[p[0], :n].copy()
        y = pop[p[1], :n].copy()
        temp = x.copy()

        n1, n2 = 0, 0
        while n1 == n2:
            n1 = np.random.randint(0, n)
            n2 = np.random.randint(0, n)
        if n1 > n2:
            n1, n2 = n2, n1

        # Perform crossover between n1 and n2
        for l in range(n1, n2 + 1):
            x[l], y[l] = y[l], temp[l]

        # Mutation
        for j in range(n):
            if np.random.rand() < p_1:
                x[j] = 1 - x[j]
            if np.random.rand() < p_1:
                y[j] = 1 - y[j]

        # Add offspring to child array
        child[k, :n] = x
        child[k, n] = obj(x)
        child[k + 1, :n] = y
        child[k + 1, n] = obj(y)

    # Replace worst m individuals with new children
    pop[:m, :] = child

    # Sort population by objective function value (ascending order)
    pop = pop[pop[:, n].argsort()]

# Print the best solution
best_solution = pop[-1, :n]
best_value = pop[-1, n]
print(f"Best solution x*: {best_solution}")
print(f"f(x*): {best_value}")


# # Question2

# In[ ]:


import numpy as np
import random

# Knapsack parameters
values = np.array([6, 8, 3, 4, 5, 9, 11, 12, 6, 8, 13, 15, 16, 13, 9, 25])
weights = np.array([3, 5, 4, 7, 4, 10, 3, 6, 8, 14, 4, 9, 10, 11, 17, 12])
capacity = 25
penalty_param = 50  # Penalty parameter R

# Randomly initialize the initial solution x0
x0 = np.array([1 if random.random() < 0.5 else 0 for _ in range(16)])

# Define the objective function F(x) with penalty
def F(x):
    total_value = np.dot(values, x)
    total_weight = np.dot(weights, x)
    penalty = max(0, total_weight - capacity)
    return total_value - penalty_param * penalty

# Fiduccia and Mattheyses (FM) algorithm
def FM_algorithm(x0):
    x = x0.copy()
    F_xmax = F(x0)
    xmax = x0.copy()
    flag = 1
    pass_num = 1

    while flag == 1:
        flag = 0
        epoch = 0
        F_set = set(range(16))  # Set of free variables (indices)
        L_set = set()           # Set of locked variables

        while epoch < 16:
            best_F_epoch = -float('inf')
            best_flip_index = -1

            # Try flipping each variable in F_set and calculate F(x)
            for j in F_set:
                x_temp = x.copy()
                x_temp[j] = 1 - x_temp[j]  # Flip the j-th variable
                F_xj = F(x_temp)

                if F_xj > best_F_epoch:
                    best_F_epoch = F_xj
                    best_flip_index = j

            # Update x by flipping the best variable found in this epoch
            if best_flip_index != -1:
                x[best_flip_index] = 1 - x[best_flip_index]
                F_set.remove(best_flip_index)
                L_set.add(best_flip_index)
                epoch += 1

            # Check if this is the best solution in this epoch
            if best_F_epoch > F_xmax:
                xmax = x.copy()
                F_xmax = best_F_epoch
                flag = 1

        # Update x0 to best solution in this epoch
        x = xmax.copy()
        pass_num += 1

    return xmax, F_xmax

# Running the FM algorithm with the initial solution
best_solution, best_value = FM_algorithm(x0)
print("Best solution x*: ", best_solution)
print("F(x*): ", best_value)


# In[23]:


import numpy as np
import random

# Knapsack parameters
values = np.array([6, 8, 3, 4, 5, 9, 11, 12, 6, 8, 13, 15, 16, 13, 9, 25])
weights = np.array([3, 5, 4, 7, 4, 10, 3, 6, 8, 14, 4, 9, 10, 11, 17, 12])
capacity = 25
penalty_param = 50  # Penalty parameter R

# Randomly initialize the initial solution x0 using the specified process
x0 = np.array([1 if random.random() < 0.5 else 0 for _ in range(16)])

# Define the objective function F(x) with penalty
def F(x):
    total_value = np.dot(values, x)
    total_weight = np.dot(weights, x)
    penalty = max(0, total_weight - capacity)
    return total_value - penalty_param * penalty

# Fiduccia and Mattheyses (FM) algorithm
def FM_algorithm(x0):
    x = x0.copy()
    F_xmax = F(x0)
    xmax = x0.copy()
    flag = 1
    pass_num = 1

    while flag == 1:
        flag = 0
        epoch = 0
        F_set = set(range(16))  # Set of free variables (indices)
        L_set = set()           # Set of locked variables

        while epoch < 16:
            best_F_epoch = -float('inf')
            best_flip_index = -1

            # Try flipping each variable in F_set and calculate F(x)
            for j in F_set:
                x_temp = x.copy()
                x_temp[j] = 1 - x_temp[j]  # Flip the j-th variable
                F_xj = F(x_temp)

                if F_xj > best_F_epoch:
                    best_F_epoch = F_xj
                    best_flip_index = j

            # Update x by flipping the best variable found in this epoch
            if best_flip_index != -1:
                x[best_flip_index] = 1 - x[best_flip_index]
                F_set.remove(best_flip_index)
                L_set.add(best_flip_index)
                epoch += 1

            # Check if this is the best solution in this epoch
            if best_F_epoch > F_xmax:
                xmax = x.copy()
                F_xmax = best_F_epoch
                flag = 1

        # Update x0 to best solution in this epoch
        x = xmax.copy()
        pass_num += 1

    return xmax, F_xmax

# Running the FM algorithm with the initial solution
best_solution, best_value = FM_algorithm(x0)
total_weight = np.dot(weights, best_solution)

# Printing results
print("Best solution x*: ", best_solution)
print("F(x*): ", best_value)
print("Total weight of the selected items: ", total_weight)


# In[ ]:




