import numpy as np


class DifferentialEvo:
    def __init__(self,
                 F: float,
                 probability_recombination: float,
                 fitness,
                 upperBound: np.ndarray,
                 lowerBound: np.ndarray,
                 num_individuals: int,
                 num_dimensions: int
                ):
        
        self.F = F
        self.probability_recombination = probability_recombination
        self.fitness = fitness
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.num_individuals = num_individuals
        self.num_dimensions = num_dimensions
        self.pop = self._initialize_population()

    def optimize(self, maximize=True):
        best_individuals = self.pop
        for parent in self.pop:
            donor = self._mutation() # iterativamente constrói a população mutada
            offspring = self._crossover(parent, donor) # resultado do cruzamento é o melhor indivíduo
            offspring = self.enforce_bounds(offspring)
            
            if maximize:
                best = self._selection_maximize(parent, offspring)
            else:
                best = self._selection_minimize(parent, offspring)
            
            parent_index = np.where((self.pop == parent).all(axis=1))[0] # seleciona o index equivalente ao do pai
            best_individuals[parent_index] = best # substitui no index equivalente ao do pai pelo melhor
        
        self.pop = best_individuals
    
    def enforce_bounds(self, offspring: np.ndarray):
        for i in range(len(offspring)):
            if offspring[i] < self.lowerBound[i]:
                offspring[i] = 2 * self.lowerBound[i] - offspring[i]
            elif offspring[i] > self.upperBound[i]:
                offspring[i] = 2 * self.upperBound[i] - offspring[i]
        return offspring
    
    def _initialize_population(self):
        rnd_distributed_individuals = np.random.rand(self.num_individuals, self.num_dimensions)
        bound_restriction = self.upperBound - self.lowerBound
        return rnd_distributed_individuals * bound_restriction + self.lowerBound
    
        
    def _mutation(self):
        xi1, xi2, xi3 = self._randomize_three_random_individuals()
        donor = xi1 + self.F*(xi2 - xi3)
        return donor
    
    def _randomize_three_random_individuals(self):
        """
        Enquanto não houver três indivíduos distintos em
        indexes, aleatorize para conseguir mais um.
        """
        
        individuals = []
        indexes = []
        
        """Apesar da complexidade elevada, não deve demorar
        muito se o tamanho da população for grande o suficiente."""
        while len(indexes) < 3:
            random_index = np.random.choice(self.pop.shape[0])
            if not random_index in indexes:
                random_individual = self.pop[random_index]
                indexes.append(random_index)
                individuals.append(random_individual)
        to_mutate = (individuals[0], individuals[1], individuals[2])
        return to_mutate

    def _crossover(self, parent: np.ndarray, donor: np.ndarray):
        """
        Recombinação binomial usando gene_donor_percentage como p_r.
        """
        n = parent.shape[0]
        offspring = parent.copy()
        
        # Gera máscara aleatória baseada em gene_donor_percentage
        mask = np.random.rand(n) < self.probability_recombination
        
        # Garante pelo menos 1 gene do doador
        if not np.any(mask):
            mask[np.random.randint(0, n)] = True
        
        offspring[mask] = donor[mask]
        return offspring

    def _selection_maximize(self, parent: np.ndarray, offspring: np.ndarray):
        if self.fitness(offspring) > self.fitness(parent):
            return offspring
        return parent
    
    def _selection_minimize(self, parent, offspring):
        if self.fitness(offspring) < self.fitness(parent):
            return offspring
        return parent
