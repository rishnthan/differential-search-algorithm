import numpy as np

# Translated from PINAR CIVICIOGLU's original code

# P.Civicioglu, "Transforming geocentric cartesian coordinates to geodetic coordinates by using differential search
# algorithm",  Computers & Geosciences, 46 (2012), 229-247.
# P.Civicioglu, "Understanding the nature of evolutionary search algorithms", Additional technical report for the
# project of 110Y309-Tubitak,2013, Ankara, Turkey.


def DSA(objectiveFunction, bounds, popSize, maxGens, F=0.5, CR=0.7):
    """
    Differential Search Algorithm (DSA) for global optimization.

    Parameters:
    - objectiveFunction: The objective function to be minimized. It should take a NumPy array as input.
    - bounds: A list of tuples, where each tuple contains the lower and upper bounds for each parameter.
    - popSize: The size of the population.
    - maxGens: The maximum number of generations.
    - F: The scaling factor for the differential mutation (default is 0.5).
    - CR: The crossover rate for the differential mutation (default is 0.7).

    Returns:
    - The best solution found by the algorithm.
    - The value of the objective function at the best solution.
    """

    dim = len(bounds)
    population = np.random.rand(popSize, dim)
    for i in range(dim):
        population[:, i] = bounds[i][0] + (bounds[i][1] - bounds[i][0]) * population[:, i]

    for generation in range(maxGens):
        for i in range(popSize):
            # Randomly select three distinct individuals
            a, b, c = np.random.choice(popSize, 3, replace=False)
            x, y, z = population[a], population[b], population[c]

            # Generate a mutant vector
            mutant = x + F * (y - z)

            # Applying crossover to create the trial vector
            j_rand = np.random.randint(dim)
            trial = np.copy(population[i])
            for j in range(dim):
                if j == j_rand or np.random.rand() < CR:
                    trial[j] = mutant[j]

            # Checking if the trial vector is better than the current individual
            if objectiveFunction(trial) < objectiveFunction(population[i]):
                population[i] = trial

    # Find the best solution in the final population
    best_solution = population[np.argmin([objectiveFunction(ind) for ind in population])]
    best_value = objectiveFunction(best_solution)

    return best_solution, best_value



# Objective Function (e.g., the sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Problem Bounds
bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # Example bounds for a 2D problem

# Algo Parameters
popSize = 50
maxGens = 100

best_solution, best_value = DSA(sphere_function, bounds, popSize, maxGens)

print("Best Solution:", best_solution)
print("Best Value:", best_value)
