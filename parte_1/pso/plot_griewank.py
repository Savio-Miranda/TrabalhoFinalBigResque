import numpy as np

from swarm import Swarm
from plot import plot_swarm

def griewank(x, y, z):
    sum_part = (x**2 + y**2 + z**2) / 4000
    prod_part = (np.cos(x/np.sqrt(1)) * 
                np.cos(y/np.sqrt(2)) * 
                np.cos(z/np.sqrt(3)))
    return sum_part - prod_part + 1

def main():
    gif_path = "gifs/griewank_PSO.gif"
    
    # Configuration
    SPACE = 1000
    NUM_PARTICLES = 30
    MAX_ITERATIONS = 50

    w = 0.8
    c1 = c2 = 0.1
    bounds = np.array([SPACE]*3), np.array([-SPACE]*3)
    alpha = 0.3
    space = {"x_min": -SPACE,
             "x_max": SPACE,
             "y_min": -SPACE,
             "y_max": SPACE,
             "resolution": 200
            }
    
    xmin = 0
    ymin = 0
    zmin = 0
    fmin = griewank(xmin, ymin, zmin) # valor mínimo conhecido

    swarm = Swarm(griewank, NUM_PARTICLES, w, c1, c2, bounds[0], bounds[1])
    
    plot_swarm(title="Griewank PSO",
               swarm=swarm,
               iterations=MAX_ITERATIONS,
               xmin=xmin,
               ymin=ymin,
               zmin=zmin,
               fmin=fmin,
               space=space,
               z_lim=(0, 400),
               alpha=alpha,
               obj=griewank,
               gif_path=gif_path)
    

if __name__ == "__main__":
    main()