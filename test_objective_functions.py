import numpy as np
from DSA import DSA

# Objective Functions 
# - Sphere Function
def sphere(x):
    return np.sum(x**2)

# - Rosenbrock Function
def rosenbrock(x):
    return np.sum((1 - x)**2 + 100 * (x - x**2)**2)

# - Booth Function
def booth(x):
    return np.sum((x + 2*3 - 7)**2 + (2*x + 3 - 5)**2)


# - Himmelblau Function
def himmelblau(x):
    return np.sum((x**2 + 3 - 11)**2 + (x + 3**2 - 7)**2)


# - Ackley Function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    return np.sum(-a * np.exp(-b * np.sqrt(0.5 * (x**2))) - np.exp(0.5 * (np.cos(c*x))) + a + np.exp(1))


best_solution, best_fitness = DSA(himmelblau, dimension=2)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
