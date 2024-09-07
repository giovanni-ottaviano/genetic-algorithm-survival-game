import argparse
import random
import numpy as np
from tabulate import tabulate

import ga_utils as gu
from individual import generate_population
from argparse_utils import positiveint, floatrange


def main():

    """
        Main function - Develop survival instinct in a population using Genetic Algorithms (version with poison).
    """

    # Get input values and set variables
    parser = argparse.ArgumentParser(description="Genetic algorithm - Survival instinct")
    parser.add_argument('dimension', help="Grid dimension", type=positiveint)
    parser.add_argument('n_individuals', help="Population size", type=positiveint)
    parser.add_argument('moves', help="Steps per generation", type=positiveint)
    parser.add_argument('--genes', help="Genome length ", type=positiveint, default=15)
    parser.add_argument('--max_generations', help="Maximum number of generations", type=positiveint, default=30)
    parser.add_argument('--energy', help="Initial energy", type=positiveint, default=10)
    parser.add_argument('--food', help="Food fraction", type=floatrange(0,1), default=0.3)
    parser.add_argument('--poison', help="Poison fraction", type=floatrange(0,1), default=0.2)
    parser.add_argument('--p_mutation', help="Probability of a mutation", type=floatrange(0,1), default=0.05)
    parser.add_argument('--randseed', help="Random seed", type=int, default=None)

    args = parser.parse_args()

    D = args.dimension
    n_individuals = args.n_individuals
    moves = args.moves
    genes = args.genes
    max_generations = args.max_generations
    init_energy = args.energy
    food_fraction = args.food
    poison_fraction = args.poison
    p_mutation = args.p_mutation
    seed = args.randseed   

    # Print warning if grid near saturation
    grid_sat_par = food_fraction + poison_fraction
    if grid_sat_par >= 1:
        raise ValueError (
            "Grid over-saturated with poison and food.",
            f"Note that 'food_fraction + poison_fraction' must be less that 1, but {grid_sat_par} was given.")
    elif grid_sat_par >= 0.85:
        print("\nWARNING: Grid near saturation. Computation could be slow!") 

    # Print input parameters
    values = [
        D,
        n_individuals,
        moves,
        genes,
        init_energy,
        food_fraction,
        poison_fraction,
        p_mutation,
        max_generations
    ] + ["None" if seed == None else seed]

    names = [
        "Grid dimension",
        "Population size",
        "Steps per generation",
        "Genes",
        "Initial energy",
        "Food fraction",
        "Poison fraction",
        "Probability of mutation",
        "Limit generation",
        "Random seed"
    ]

    appo_tabulate = [[name, val] for name, val in zip(names, values)]

    print("Genetic Algorithm to develop survival instinct - Input parameters:")
    print("...")
    print(tabulate(appo_tabulate, headers=["Parameter", "Value"], tablefmt="psql"))
    print("...")
    print("...")

    # Initialization
    random.seed(seed)
    grid = np.zeros((D,D), dtype=np.short)
    gu.distribute_food(grid, food_fraction)
    gu.distribute_poison(grid, poison_fraction)

    population = generate_population(n_individuals, init_energy, genes, D)


    # Run simulation (while at least one individual is alive or max number of genertion is reached)
    index_generations = 0
    learning_pct, learning_poison_pct, mean_energy, max_energy_ind = [], [], [], []
    while gu.is_population_alive(population) and index_generations < max_generations:
        # Check the learning process
        learning_pct.append(gu.learning_percentage(population))
        learning_poison_pct.append(gu.poison_percentage(population))
        mean_energy.append(gu.avg_energy(population))
        max_energy_ind.append(gu.max_energy_individual(population).energy)

        for _ in range(moves):
            for ind in population:
                ind.move(grid)

        population = gu.update_population(population, p_mutation)
        gu.distribute_food(grid, food_fraction)
        gu.distribute_poison(grid, poison_fraction)

        index_generations += 1

    # Print results in tabular form
    final_values = [
        "{:.1f}".format(learning_pct[0]),
        "{:.1f}".format(learning_pct[-1]),
        "{:.1f}".format(learning_poison_pct[0]),
        "{:.1f}".format(learning_poison_pct[-1]),
        "{:.1f}".format(mean_energy[0]),
        "{:.1f}".format(mean_energy[-1]),
        "{:.1f}".format(max_energy_ind[0]),
        "{:.1f}".format(max_energy_ind[-1]),
    ]

    final_names = [
        "Learning food percentage (start)",
        "Learning food percentage (end)",
        "Learning poison percentage (start)",
        "Learning poison percentage (end)",
        "Average energy (start)",
        "Average energy (end)",
        "Max energy individual (start)",
        "Max energy individual (end)",
    ]

    appo_tabulate = [[name, val] for name, val in zip(final_names, final_values)]

    print(f"Results after {index_generations} generations")
    print("...")
    print(tabulate(appo_tabulate, headers=["Parameter", "Value"], tablefmt="psql"))
    print("...")

    return 0


if __name__ == '__main__':
    main()