import matplotlib.pyplot as plt

from genetics import *


def genetic_algorithm_with_plotting():
    population = ini_population(POPULATION_SIZE, GENOME_LENGTH)
    best_solution = None
    best_fitness_ever = float('-inf')
    
    # Listas para armazenar dados para plotagem
    generation_numbers = []
    best_fitnesses = []
    average_fitnesses = []
    current_fitnesses = []

    for generation in range(GENERATIONS):
        fitness_values = [fitness(genome) for genome in population]
        current_best = max(fitness_values)
        current_avg = sum(fitness_values) / len(fitness_values)
        
        current_fitnesses.append(current_best)
        average_fitnesses.append(current_avg)
        generation_numbers.append(generation)

        if current_best > best_fitness_ever:
            best_fitness_ever = current_best
            best_index = fitness_values.index(current_best)
            best_solution = population[best_index].copy()

            # Armazenar dados para plotagem
            best_fitnesses.append(current_best)
        else:
            best_fitnesses.append(best_fitness_ever)

        
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = selection_parents(population, fitness_values)
            parent2 = selection_parents(population, fitness_values)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        
        population = new_population
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness_ever}")


    # Plotando os resultados
    plt.figure(figsize=(12, 6))
    plt.plot(generation_numbers, best_fitnesses, label='Melhor Aptidão', color='blue')
    plt.plot(generation_numbers, current_fitnesses, label='Aptidão atual', color='pink', alpha=0.5)
    plt.plot(generation_numbers, average_fitnesses, label='Aptidão Média', color='green', alpha=0.5)
    
    plt.title('Progresso do Algoritmo Genético')
    plt.xlabel('Geração')
    plt.ylabel('Aptidão')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Destacar a melhor solução encontrada
    max_fitness = max(best_fitnesses)
    max_gen = best_fitnesses.index(max_fitness)
    plt.scatter(max_gen, max_fitness, color='red', zorder=5, label=f'Melhor: {max_fitness:.2f}')
    plt.annotate(f'{max_fitness:.2f}', (max_gen, max_fitness), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nFinal Results:")
    print_solution(best_solution)
    print(f"Best fitness achieved: {best_fitness_ever}")

# Para executar o algoritmo com plotagem:
if __name__ == "__main__":
    genetic_algorithm_with_plotting()