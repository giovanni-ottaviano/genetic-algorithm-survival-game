import random
from math import ceil
from collections import Counter
from typing_extensions import Self
import numpy as np


class individual():

    """
        Class individual -> Member of the population.
        
        Each individual is specified by:
        - position on the grid (x,y)
        - energy
        - genes (binary)
        - dimension of the grid D
    """

    def __init__(self, energy: int, coord: tuple, D: int, genes: list=[]) -> None:
        self.energy = energy
        self.x, self.y = coord
        self.D = D
        self.genes = genes


    def mate(self, other: Self, null_energy: bool=False) -> list:

        """
            Returns a list with two individuals after selected crossover operation.
            The new individual has the following properties:
            - position on the grid: random
            - energy: average (rounded up) from parents or zero (if null_energy = True)
            - genes: recombined via crossover

            This method acts on self and another given individual.

        """    

        offsprings_genes = crossover(self.genes, other.genes)
        offsprings = []
        
        for i in range(2):
            x = random.randint(0, self.D - 1)
            y = random.randint(0, self.D - 1)
            energy = ceil((self.energy + other.energy)/2) if not null_energy else 0

            # Add new individual
            offsprings.append(individual(energy, (x,y), self.D, offsprings_genes[i]))

        return offsprings

    def mutate(self, p_mutation: float) -> None:

        """
            Mutate individual with given probability.

            Possible mutations:
            - Bit swap
        """
        
        if random.random() < p_mutation:
            possible_genes = [0,1]
            index = random.randint(0, len(self.genes) - 1)
            possible_genes.remove(self.genes[index])

            self.genes[index] = random.choice(possible_genes)


    def update_energy(self, grid: np.ndarray) -> None:

        """
            Update energy after making a step on the grid.
            Each step costs energy (-1), but finding food on the grid can partially restore energy.
        
            Check the grid: 
            0 -> no food (+0)
            1 -> food (+2)
        """

        # Spend energy for current step
        self.energy -= 1

        # Look for food (eat and restore energy)
        if grid[self.x, self.y] == 1:
            grid[self.x, self.y] = 0
            self.energy += 2

    def step(self, direction: int) -> None:

        """Make step in given direction (clockwise)"""

        if direction == 0:
            self.y = pbc(self.y + 1, self.D)
        elif direction == 1:
            self.x = pbc(self.x + 1, self.D)
        elif direction == 2:
            self.y = pbc(self.y - 1, self.D)
        elif direction == 3:
            self.x = pbc(self.x - 1, self.D)
        else:
            raise ValueError ("Wrong direction in method 'step':", direction)


    def move(self, grid: np.ndarray) -> None:

        """
            Move the individual on the grid (one step, only if it'alive) and update its energy.

            If energy > 0, make a step based on the nearest positions of the grid:
            1) Identical spot: choose random direction
            2) At least one spot different from the others: choose a random direction, using genes as weights

        """

        if self.energy > 0:
            neighbours = []

            # Append in clockckwise order (up-right-down-left)
            for i in [1,-1]:
                neighbours.append(grid[pbc(self.x, self.D), pbc(self.y + i, self.D)])
                neighbours.append(grid[pbc(self.x + i, self.D), pbc(self.y, self.D)])

            #check if all directions are the same
            if len(set(neighbours)) == 1:
                self.step(random.randint(0,3))
            else:
                #make weights for rigged wheel
                count_genes = Counter(self.genes)
                list_genes, weights = [], []

                for gene, count in count_genes.items():
                    list_genes.append(gene)
                    weights.append(count)

                #N.B: if two directions are the same, .index chooses always the first occurrence
                type_chosen = random.choices(list_genes, weights=weights, k=1)
                direction = neighbours.index(type_chosen)

                self.step(direction)

            self.update_energy(grid)

# Binary crossover
def crossover(parent1: list, parent2: list) -> list:

    '''
        Random Respectful Crossover (RRC)

        Offspings are generated according to the values of a "similarity vector",
        which contains 1 (or 0) as i-th gene if both parents have the same i-th gene.
        Other genes are selected via uniform random sampling.
    
    '''

    if len(parent1) != len(parent2):
        raise ValueError ("Dimension mismatch in selected parents: {} and {}".format(len(parent1), len(parent2)))


    # Generate similarity vector
    similarity_vector = [gene1 if gene1 == gene2 else None for gene1, gene2 in zip(parent1, parent2)]
    offsprings = [similarity_vector[:], similarity_vector[:]]

    # Select other genes with uniform probability
    for offspring in offsprings:
        for i in range(len(offspring)):
            if offspring[i] is None:
                offspring[i] = 1 if random.random() > 0.5 else 0

    return offsprings

def generate_population(n_pop: int, energy: int, n_genes: int, D: int) -> list:

    """
        Returns a population of properly initialized individuals
    """

    population = []

    for _ in range(n_pop):
        x = random.randint(0, D-1)
        y = random.randint(0, D-1)
        genes = random.choices([0,1], k=n_genes)

        population.append(individual(energy, (x,y), D, genes))

    return population


def pbc(i: int, D: int) -> int:

    '''Periodic Boundary Conditions'''

    return i % D