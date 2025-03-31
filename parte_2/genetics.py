import random

# Constants for genetic algorithm
POPULATION_SIZE = 400  # Number of individuals in each generation
MUTATION_RATE = 0.25    # Probability of mutating a gene
CROSSOVER_RATE = 0.5    # Probability of performing crossover
GENERATIONS = 310      # Number of generations to run the algorithm

# Example knapsack problem parameters
ITEMS = [  # List of available items (value, weight) pairs
    (60, 10.53), (10, 19.11111), (10, 32), (68, 32.455), (10, 9.5),
    (70, 29.98), (19, 33.65), (30, 20), (10, 12.2), (15.115426, 12),
    (10, 16), (10, 31), (20, 1.0), (14, 22), (12, 19.5), (10, 33),
    (10, 8), (3, 1.5), (10, 6)
    ]
GENOME_LENGTH = len(ITEMS)  # Number of items available
MAX_WEIGHT = 110  # Knapsack weight capacity

# Generate a random genome (binary representation of item selection)
def random_genome(length):
    return [random.randint(0, 1) for _ in range(length)]  # 0 means not selected, 1 means selected

# Initialize a population with random genomes
def ini_population(population_size, genome_length):
    return [random_genome(genome_length) for _ in range(population_size)]  # Create multiple random genomes



# Compute fitness of a genome
def fitness(genome):
    total_value = 0  # Total value of selected items
    total_weight = 0  # Total weight of selected items
    
    # Iterate through genome to calculate total value and weight
    for i, selected in enumerate(genome):
        if selected:  # If item is selected
            value, weight = ITEMS[i]  # Retrieve value and weight
            total_value += value  # Accumulate value
            total_weight += weight  # Accumulate weight
    
    # Penalize solutions that exceed the weight limit
    if total_weight > MAX_WEIGHT:
        excess = total_weight - MAX_WEIGHT
        return -excess  # Negative penalty for exceeding weight
    return total_value  # Higher value is better



# Select a parent using a fitness-based selection (roulette wheel method)
def selection_parents(population, fitness_values):
    min_fitness = min(fitness_values)  # Find the lowest fitness value
    adjusted_fitness = [f + min_fitness + 1 for f in fitness_values]  
    
    total_fitness = sum(adjusted_fitness)  # Sum of all fitness values
    pick = random.uniform(0, total_fitness)  # Random pick in the fitness range
    current = 0  # Cumulative sum tracker
    for individual, fitness_value in zip(population, adjusted_fitness):
        current += fitness_value
        if current > pick:
            return individual  # Select the corresponding individual
    return population[-1]  # Fallback in case of rounding issues

# Perform crossover between two parents to create offspring
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:  # Apply crossover with given probability
        crossover_point = random.randint(1, GENOME_LENGTH - 1)  # Choose a random point
        child1 = parent1[:crossover_point] + parent2[crossover_point:]  # First offspring
        child2 = parent1[crossover_point:] + parent2[:crossover_point]  # Second offspring
        return child1, child2
    else:
        return parent1, parent2  # No crossover, return parents unchanged

# Apply mutation to a genome (random bit flip)
def mutate(genome):
    for i in range(len(genome)):
        if random.random() < MUTATION_RATE:  # Check mutation probability
            genome[i] = abs(genome[i] - 1)  # Flip bit (0 -> 1, 1 -> 0)
    return genome

# Print the best solution found
def print_solution(genome):
    total_value = 0  # Total value of selected items
    total_weight = 0  # Total weight of selected items
    selected_items = []  # List of selected item indices
    
    # Iterate through genome to determine selected items
    for i, selected in enumerate(genome):
        if selected:
            value, weight = ITEMS[i]
            total_value += value
            total_weight += weight
            selected_items.append(i)  # Store selected item index
    
    print(f"Selected items (indices): {selected_items}")
    print(f"Total value: {total_value}")
    print(f"Total weight: {total_weight}/{MAX_WEIGHT}")

# Main genetic algorithm function
def genetic_algorithm():
    population = ini_population(POPULATION_SIZE, GENOME_LENGTH)  # Initialize population
    best_solution = None  # Store best solution found
    best_fitness_ever = float('-inf')  # Track highest fitness value
    
    # Run genetic algorithm for a defined number of generations
    for generation in range(GENERATIONS):
        fitness_values = [fitness(genome) for genome in population]  # Evaluate fitness
        current_best = max(fitness_values)  # Find the best fitness in this generation
        
        # Track the best solution found so far
        if current_best > best_fitness_ever:
            best_fitness_ever = current_best
            best_index = fitness_values.index(current_best)
            best_solution = population[best_index].copy()  # Store best genome
        
        # Create next generation
        new_population = []
        for _ in range(POPULATION_SIZE // 2):  # Each iteration generates two offspring
            parent1 = selection_parents(population, fitness_values)  # Select first parent
            parent2 = selection_parents(population, fitness_values)  # Select second parent
            offspring1, offspring2 = crossover(parent1, parent2)  # Perform crossover
            new_population.extend([mutate(offspring1), mutate(offspring2)])  # Apply mutation and add to new population
        
        population = new_population  # Replace old population with new generation
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness_ever}")
    
    # Display final best solution
    print("\nFinal Results:")
    print_solution(best_solution)
    print(f"Best fitness achieved: {best_fitness_ever}")

# Run the genetic algorithm if script is executed directly
if __name__ == "__main__":
    genetic_algorithm()
