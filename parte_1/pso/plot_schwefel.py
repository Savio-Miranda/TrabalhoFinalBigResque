import numpy as np

from swarm import Swarm
from plot import plot_swarm

def schwefel(x, y, z):
    x_sin = x * np.sin(np.sqrt(np.abs(x)))
    y_sin = y * np.sin(np.sqrt(np.abs(y)))
    z_sin = z * np.sin(np.sqrt(np.abs(z)))
    return (418.9829 * 3) - (x_sin + y_sin + z_sin)
    

def main():
    gif_path = "gifs/schwefel_PSO.gif"
    
    # Configuration
    SPACE = 500
    NUM_PARTICLES = 30
    MAX_ITERATIONS = 100

    w = 0.9
    c1 = c2 = 0.4
    bounds = np.array([SPACE]*3), np.array([-SPACE]*3)
    alpha = 0.3
    space = {"x_min": -SPACE,
             "x_max": SPACE,
             "y_min": -SPACE,
             "y_max": SPACE,
             "resolution": 400
            }
    
    xmin = 420.9687
    ymin = 420.9687
    zmin = 420.9687
    fmin = schwefel(xmin, ymin, zmin) # valor mínimo conhecido

    swarm = Swarm(schwefel, NUM_PARTICLES, w, c1, c2, bounds[0], bounds[1])
    
    plot_swarm(title="Schwefel PSO",
               swarm=swarm,
               iterations=MAX_ITERATIONS,
               xmin=xmin,
               ymin=ymin,
               zmin=zmin,
               fmin=fmin,
               space=space,
               z_lim=(0, 1300),
               alpha=alpha,
               obj=schwefel,
               gif_path=gif_path)
    

if __name__ == "__main__":
    main()