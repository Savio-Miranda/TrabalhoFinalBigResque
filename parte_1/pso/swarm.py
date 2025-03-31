import numpy as np
from typing import Tuple
from particle import Particle

class Swarm:
    def __init__(self, fitness, num_particles: int, w: float, c1: float, c2: float,
                 upperBound: np.ndarray, lowerBound: np.ndarray):
        self.fitness = fitness
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.bounds = (upperBound, lowerBound)
        
        self.particles = None
        self.best_global_position = None
        self.best_global_value = float('inf')
        self._initialize_swarm()
        self._initialize_global_best_position_and_value()

    def _initialize_swarm(self) -> list[Particle]:
        pos_components = np.random.uniform(self.lowerBound, self.upperBound, (self.num_particles, 3))
        vel_components = np.random.uniform(-1, 1, (self.num_particles, 3)) * 0.1
        particles = []
        for i in range(self.num_particles):
            particle = Particle(pos_components[i], vel_components[i])
            particle.best_value = self.fitness(*particle.position)
            particles.append(particle)
        self.particles = particles
        return

    def _initialize_global_best_position_and_value(self):
        for p in self.particles:
            current_value = self.fitness(*p.position)
            if current_value < self.best_global_value:
                self.best_global_value = current_value
                self.best_global_position = p.position.copy()
        return

    def optimize(self):
        self._update_particles()
        return

    def _update_particles(self):
        for p in self.particles:
            p.update_velocity(self.w, self.c1, self.c2, self.best_global_position)
            p.update_position(self.bounds)
            
            # Avalia nova posição
            current_value = self.fitness(*p.position)
            
            # Atualiza melhor posição individual
            if current_value < p.best_value:
                p.best_value = current_value
                p.best_position = p.position.copy()
                
                # Atualiza melhor posição global
                if current_value < self.best_global_value:
                    self.best_global_value = current_value
                    self.best_global_position = p.position.copy()
        return

    def get_swarm_status(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions = np.array([p.position for p in self.particles]).T
        velocities = np.array([p.velocity for p in self.particles]).T
        best_positions = np.array([p.best_position for p in self.particles]).T
        global_best = self.best_global_position.reshape(3, 1)
        
        return positions, velocities, best_positions, global_best