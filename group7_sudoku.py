import random
import numpy as np
import matplotlib.pyplot as plt
import time

# Class representing a Sudoku puzzle
class Sudoku:
    def __init__(self, board):
        self.board = board

    def is_valid(self):
        # Check each row, column, and 3x3 sub-grid for duplicates
        for i in range(9):
            row = [self.board[i][j] for j in range(9) if self.board[i][j] != 0]
            col = [self.board[j][i] for j in range(9) if self.board[j][i] != 0]
            if len(set(row)) != len(row) or len(set(col)) != len(col):
                return False
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                block = [self.board[m][n] for m in range(i, i + 3) for n in range(j, j + 3) if self.board[m][n] != 0]
                if len(set(block)) != len(block):
                    return False
        return True

# Class representing an individual in the population
class Individual:
    def __init__(self, board, generation=0):
        self.board = np.copy(board)
        self.generation = generation
        self.fitness = 0
        self.fill_zeros()

    def fill_zeros(self):
        # Fill all zero cells with random numbers between 1 and 9
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    self.board[i][j] = random.randint(1, 9)

    def calculate_fitness(self):
        # Calculate fitness based on the number of unique numbers in rows, columns, and sub-grids
        unique_numbers = 0
        for i in range(9):
            unique_numbers += len(set(self.board[i]))  # row
            unique_numbers += len(set([self.board[j][i] for j in range(9)]))  # column
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                block = [self.board[m][n] for m in range(i, i + 3) for n in range(j, j + 3)]
                unique_numbers += len(set(block))
        self.fitness = unique_numbers
        return self.fitness

    # Implement cycle crossover
    def cycle_xo(self, other):
        # Convert the boards to one-dimensional lists
        p1 = [cell for row in self.board for cell in row]
        p2 = [cell for row in other.board for cell in row]

        # Offspring placeholders
        offspring1 = [None] * len(p1)
        offspring2 = [None] * len(p1)

        while None in offspring1:
            index = offspring1.index(None)
            val1 = p1[index]
            val2 = p2[index]

            # Copy the cycle elements
            while val1 != val2:
                offspring1[index] = p1[index]
                offspring2[index] = p2[index]
                val2 = p2[index]
                index = p1.index(val2)

            # Copy the rest
            for element in offspring1:
                if element is None:
                    index = offspring1.index(None)
                    if offspring1[index] is None:
                        offspring1[index] = p2[index]
                        offspring2[index] = p1[index]

        # Convert the one-dimensional lists back to boards
        offspring1_board = [offspring1[i:i + 9] for i in range(0, len(offspring1), 9)]
        offspring2_board = [offspring2[i:i + 9] for i in range(0, len(offspring2), 9)]

        return Individual(offspring1_board), Individual(offspring2_board)

    # Implement single-point crossover
    def single_point_xo(self, parent1, parent2):
        # Convert the boards to one-dimensional lists
        p1 = [cell for row in parent1.board for cell in row]
        p2 = [cell for row in parent2.board for cell in row]

        xo_point = random.randint(1, len(p1) - 1)
        offspring1 = p1[:xo_point] + p2[xo_point:]
        offspring2 = p2[:xo_point] + p1[xo_point:]

        # Convert the one-dimensional lists back to boards
        offspring1_board = [offspring1[i:i + 9] for i in range(0, len(offspring1), 9)]
        offspring2_board = [offspring2[i:i + 9] for i in range(0, len(offspring2), 9)]

        return Individual(offspring1_board), Individual(offspring2_board)

    # Implement uniform crossover
    def uniform_xo(self, other):
        child1_board = np.copy(self.board)
        child2_board = np.copy(other.board)
        for i in range(9):
            for j in range(9):
                if random.random() < 0.5:
                    child1_board[i][j] = other.board[i][j]
                    child2_board[i][j] = self.board[i][j]
        return [Individual(child1_board), Individual(child2_board)]

    def crossover(self, other_individual, crossover_strategy):
        # Choose crossover strategy
        if crossover_strategy == 'single_point':
            return self.single_point_xo(self, other_individual)
        elif crossover_strategy == 'uniform':
            return self.uniform_xo(other_individual)
        elif crossover_strategy == 'cycle':
            return self.cycle_xo(other_individual)
        else:
            raise ValueError("Invalid crossover strategy")

    # Implement swap mutation
    def swap_mutation(self):
        # Choose two random positions on the board
        row1, col1, row2, col2 = random.randint(0, 8), random.randint(0, 8), random.randint(0, 8), random.randint(0, 8)
        # Swap the elements at these positions
        self.board[row1][col1], self.board[row2][col2] = self.board[row2][col2], self.board[row1][col1]
        return self

    # Implement scramble mutation
    def scramble_mutation(self):
        start = random.randint(0, 8)
        end = random.randint(start, 8)
        for i in range(9):
            subset = self.board[i][start:end+1]
            random.shuffle(subset)
            self.board[i][start:end+1] = subset
        return self

    # Implement inversion mutation
    def inversion_mutation(self):
        start = random.randint(0, 8)
        end = random.randint(start, 8)
        for i in range(9):
            self.board[i][start:end+1] = self.board[i][start:end+1][::-1]
        return self

    def mutation(self, mutation_rate, mutation_strategy):
        # Choose mutation strategy
        if mutation_strategy == 'inversion':
            return self.inversion_mutation()
        elif mutation_strategy == 'scramble':
            return self.scramble_mutation()
        elif mutation_strategy == 'swap':
            return self.swap_mutation()
        else:
            raise ValueError("Invalid mutation strategy")

# Class representing the genetic algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, board_initial):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_solution = None
        self.solutions_list = []
        self.board_initial = board_initial

    # Initialize the population with random individuals
    def initialize_population(self):
        for i in range(self.population_size):
            self.population.append(Individual(self.board_initial))

    # Sort population based on fitness
    def sort_population(self):
        self.population = sorted(self.population, key=lambda individual: individual.calculate_fitness(), reverse=True)

    # Update the best solution found
    def best_individual(self, individual):
        if not self.best_solution or individual.fitness > self.best_solution.fitness:
            self.best_solution = individual

    # Calculate the sum of fitness values in the population
    def sum_evaluations(self):
        return sum(individual.fitness for individual in self.population)

    # Roulette wheel selection (not explicitly used in solve method)
    def select_parent(self, sum_evaluation):
        parent = -1
        value_random = random.random() * sum_evaluation
        sum_values = 0
        i = 0
        while i < len(self.population) and sum_values < value_random:
            sum_values += self.population[i].fitness
            parent += 1
            i += 1
        return parent

    # Tournament selection
    def tournament_selection(self, tournament_size):
        selected_parents = []
        for _ in range(self.population_size):
            participants = random.sample(self.population, tournament_size)
            winner = max(participants, key=lambda x: x.fitness)
            selected_parents.append(winner)
        return selected_parents

    # Rank selection
    def rank_selection(self):
        sorted_population = sorted(self.population, key=lambda x: x.fitness)
        total_ranks = self.population_size * (self.population_size + 3) / 2
        probabilities = [(i + 1) / total_ranks for i in range(self.population_size)]
        selected_parents = np.random.choice(sorted_population, self.population_size, p=probabilities)
        return selected_parents.tolist()

    def solve(self, mutation_rate, number_generations, crossover_strategy, mutation_strategy,
              selection_method='tournament', tournament_size=5, elitism=False, elite_size=0):
        self.initialize_population()

        for generation in range(number_generations):
            new_population = []

            if elitism:
                # Sort the population in descending order of fitness
                self.population.sort(key=lambda x: x.calculate_fitness(), reverse=True)
                # Select the top 'elite_size' individuals
                elites = self.population[:elite_size]
                # Include these elites in the next generation
                new_population.extend(elites)

            while len(new_population) < self.population_size:
                if selection_method == 'tournament':
                    parents = self.tournament_selection(tournament_size)
                elif selection_method == 'rank':
                    parents = self.rank_selection()
                else:
                    raise ValueError("Invalid selection method")

                parent1, parent2 = parents[0], parents[1]

                offspring = parent1.crossover(parent2, crossover_strategy)

                new_population.extend([offspring[0].mutation(mutation_rate, mutation_strategy),
                                       offspring[1].mutation(mutation_rate, mutation_strategy)])

            self.population = new_population

            for individual in self.population:
                individual.calculate_fitness()

            self.sort_population()

            best = self.population[0]
            self.solutions_list.append(best.fitness)
            self.best_individual(best)

            # Print the best solution of the current generation
            print(f"Generation {generation + 1}: Best Fitness = {best.fitness}")

            if best.fitness == 243:
                print(f"Generation {generation + 1}: Best Fitness = {best.fitness}")
                break

        return self.best_solution.board

# Define multiple Sudoku boards with varying difficulty levels
easy_board = [
    [2, 0, 5, 0, 0, 9, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 3, 0, 7],
    [7, 0, 0, 8, 5, 6, 0, 1, 0],
    [4, 5, 0, 7, 0, 0, 1, 0, 0],
    [0, 0, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 8, 5],
    [0, 2, 0, 4, 1, 8, 0, 0, 6],
    [6, 0, 8, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 2, 0, 0, 7, 0, 8]
]

medium_board = [
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0]
]

hard_board = [
    [0, 0, 0, 0, 0, 0, 0, 1, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 7, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 4, 0, 7],
    [0, 0, 0, 0, 6, 0, 0, 0, 0],
    [9, 0, 3, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 8, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5],
    [7, 2, 0, 0, 0, 0, 0, 0, 0]
]

boards = {'Easy': easy_board, 'Medium': medium_board, 'Hard': hard_board}

# Parameters for the Genetic Algorithm
population_size = 300
mutation_rate = 0.2
number_generations = 1000
crossover_strategy = 'single_point'
mutation_strategy = 'swap'
selection_method = 'tournament'
tournament_size = 8
elitism = False
elite_size = 15

results = []

for difficulty, board in boards.items():
    generations_to_optimal = []
    for _ in range(10):
        ga = GeneticAlgorithm(population_size, board)
        ga.solve(mutation_rate, number_generations, crossover_strategy, mutation_strategy,
                 selection_method, tournament_size, elitism, elite_size)
        if ga.best_solution and ga.best_solution.fitness == 243:
            generations_to_optimal.append(ga.generation)
        else:
            generations_to_optimal.append(number_generations)
    results.append((difficulty, generations_to_optimal))

# Plot results
for result in results:
    difficulty, generations = result
    plt.plot(generations, label=difficulty)

plt.title("Generations to Reach Fitness 243")
plt.xlabel("Run Number")
plt.ylabel("Generations")
plt.legend()
plt.show()