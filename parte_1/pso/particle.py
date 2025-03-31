import numpy as np


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self._position = position
        self._velocity = velocity
        self._best_position = position.copy()
        self._best_value = float('inf')
    
    def update_velocity(self, w: float, c1: float, c2: float, global_position: np.ndarray):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self._best_position - self._position)
        social = c2 * r2 * (global_position - self._position)
        self._velocity = w * self._velocity + cognitive + social
    
    def update_position(self, bounds):
        self._position = self._position + self._velocity
        # Confinamento dentro dos limites
        self._position = np.clip(self._position, bounds[1], bounds[0])
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
    
    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        self._velocity = value
    
    @property
    def best_position(self):
        return self._best_position
    
    @best_position.setter
    def best_position(self, value):
        self._best_position = value
    
    @property
    def best_value(self):
        return self._best_value
    
    @best_value.setter
    def best_value(self, value):
        self._best_value = value
    
    def __str__(self):
        return f"Position: {self._position} - Velocity: {self._velocity}"