import numpy as np

def DSA(objective_function, dimension, population_size=20, max_iterations=100,
                                   scaling_factor=0.5, crossover_probability=0.7, lower_bound=-5.12, upper_bound=5.12):
    """
    Differential Search Algorithm (DSA) for optimization.

    Parameters:
    - objective_function: Function to be optimized. It should take a numpy array of size 'dimension' as input.
    - dimension: Dimensionality of the search space.
    - population_size: Number of individuals in the population.
    - max_iterations: Maximum number of iterations.
    - scaling_factor: Scaling factor for mutation.
    - crossover_probability: Crossover probability for recombination.
    - lower_bound: Lower bound of the search space.
    - upper_bound: Upper bound of the search space.
    """

    # Initialize  population with random solutions
    population = np.random.uniform(lower_bound, upper_bound, size=(population_size, dimension))

    # Iterate to max iterations
    for iteration in range(max_iterations):
        for i in range(population_size):
            # Select three distinct individuals from the population
            while True:
                a, b, c = np.random.choice(population_size, 3, replace=False)
                if a != b and b != c and a != c:
                    break

            # Mutation
            mutant = population[a] + scaling_factor * (population[b] - population[c])

            # Ensure mutant is within bounds
            mutant = np.clip(mutant, lower_bound, upper_bound)

            # Crossover
            crossover_mask = np.random.rand(dimension) < crossover_probability
            trial_solution = np.where(crossover_mask, mutant, population[i])

            # Evaluate the trial solution
            if objective_function(trial_solution) < objective_function(population[i]):
                population[i] = trial_solution

        # Get best solution in the current population
        best_solution = population[np.argmin([objective_function(x) for x in population])]
        best_fitness = objective_function(best_solution)

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness
