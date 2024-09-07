import random
from math import floor
import numpy as np


def compute_fitness(population: list) -> list:

    """Resturns a list with the fitness of each individual (energy)"""

    fitness = [ind.energy for ind in population]

    return fitness


# Standard version: crossover with p=1, no elitism, no mutations
def update_population(pop: list, p_mutation: float) -> list:

    """
        Generate a new population from list of individuals using individual's mate method.
        Selection is based on the fitness function (here, individual's energy).

        Returned population has the same size as the original one.
    """

    fitness = compute_fitness(pop)
    new_pop = []

    n_xover = len(pop) // 2  # Number of crossovers

    # Loophole in weighted_sample due to problem specifications
    null_energy = False if np.count_nonzero(fitness) >= 2 else True

    # Crossover
    for _ in range(n_xover):
        parent1, parent2 = weighted_sample(pop, fitness, 2)
        new_pop.extend(parent1.mate(parent2, null_energy))

    # If odd, add another element
    if len(pop) % 2 == 1:
        parent1, parent2 = weighted_sample(pop, fitness, 2)
        new_pop.append(parent1.mate(parent2, null_energy)[0])

    # Apply mutation on each individual
    for ind in new_pop:
        ind.mutate(p_mutation)

    return new_pop


def is_population_alive(population: list) -> bool:

    """Check if at least one individual is alive in given population."""

    is_alive = any(ind.energy > 0 for ind in population)

    return is_alive


def distribute_food(grid: np.ndarray, food_fraction: float) -> None:

    """
        Distribute new food supply on the grid, after the residual food is removed.

        Food_fraction in (0,1) represent the fraction of positions with food on the grid.
    """

    if food_fraction <= 0 or food_fraction >= 1: 
        raise ValueError(f"Food fraction on the grid must be in (0,1), but {food_fraction} was given.")

    # Remove food from grid
    grid.fill(0)

    # Add food
    n_food = floor(grid.size * food_fraction)
    food_pos = random.sample(range(grid.size), k=n_food)

    for pos in food_pos:
        row = pos // grid.shape[0]
        col = pos % grid.shape[1]

        grid[row, col] = 1


def distribute_poison(grid: np.ndarray, poison_fraction: float) -> None:

    """
        Distribute new poison supply on the grid, avoiding positions filled with food.

        Poison_fraction in (0,1) represent the fraction of positions on the grid with poison.

        This function should be called after a cleaning operation (here implemented in 'distribute_food')

    """

    if poison_fraction <= 0 or poison_fraction >= 1: 
        raise ValueError(f"Poison fraction on the grid must be in (0,1), but {poison_fraction} was given.")


    # Add poison
    n_poison = floor(grid.size * poison_fraction)
    index = 0

    while index < n_poison:
        pos = random.randrange(grid.size)
        row = pos//grid.shape[0]
        col = pos%grid.shape[1]

        if grid[row, col] == 0:
            grid[row, col] = -1
            index += 1


def learning_percentage(population: list) -> float:

    """Returns the percentage of 'Food' (1) coded in individual's genes."""

    n_food = 0
    n_genes = len(population[0].genes)

    for ind in population:
        n_food += ind.genes.count(1)

    return 100 * n_food / (n_genes * len(population))


def poison_percentage(population: list) -> float:

    """Returns the percentage of 'Poison' (-1) coded in individual's genes."""

    n_poison = 0
    n_genes = len(population[0].genes)

    for ind in population:
        n_poison += ind.genes.count(-1)

    return 100 * n_poison / (n_genes * len(population))


def neutral_percentage(population: list) -> float:

    """Returns the percentage of 'Neutral' (0) coded in individual's genes."""

    n_neutral = 0
    n_genes = len(population[0].genes)

    for ind in population:
        n_neutral += ind.genes.count(0)

    return 100 * n_neutral / (n_genes * len(population))    


def max_energy_individual(population: list) -> float:

    """Returns the energy of the fittest individual."""

    appo_energy = [ind.energy for ind in population]
    max_index = appo_energy.index(max(appo_energy))

    return population[max_index]


def min_energy_individual(population: list) -> float:

    """Returns the energy of the least fit individual."""

    appo_energy = [ind.energy for ind in population]
    min_index = appo_energy.index(min(appo_energy))

    return population[min_index]


def avg_energy(population: list) -> float:

    """Returns the average energy for given population."""

    list_energy = [ind.energy for ind in population]

    return np.mean(list_energy, dtype=float)


def std_energy(population: list) -> float:

    """Returns the standard deviation of energy for given population."""

    list_energy = [ind.energy for ind in population]

    return np.std(list_energy, dtype=float, ddof=0)


# Same behavior as: numpy.random.choice(a, size=None, replace=False, p=None), but using module random only
def weighted_sample(v: list, scores: list, n: int) -> list:

    """
        Generates a random weighted sample (without replacement) from a given iterable.

        This version contains a loophole. Sometimes just one individual of the population is alive (energy > 0) and this function cannot
        select a second individual for mating process. For this reason, if the total weight is zero, a random (uniform) individual is selected.

    """

    if n < 0 or n > len(v) or not isinstance(n, int):
        raise ValueError ("{} not valid. It must be integer in (0, {})".format(n, len(v)))

    # Copy initial list and scores
    chosen = []
    v_c = v[:]
    scores_c = scores[:]

    
    for i in range(n):
        totscore = sum(scores_c)

        # Loophole
        if totscore == 0.:
            i = random.randrange(len(v_c))

            chosen.append(v_c.pop(i))
            scores_c.pop(i)
        else:    
            r = random.random()
            tot = 0

            for j, s in enumerate(scores_c):
                tot += s / float(totscore)

                if tot > r:
                    break

            chosen.append(v_c.pop(j))
            scores_c.pop(j)

    return chosen