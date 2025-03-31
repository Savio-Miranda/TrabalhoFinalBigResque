import numpy as np
import matplotlib.pyplot as plt

from de import DifferentialEvo
from plot_test import plot_de


def griewank(x, y, z):
        sum_part = (x**2 + y**2 + z**2) / 4000
        prod_part = (np.cos(x/np.sqrt(1)) * 
                     np.cos(y/np.sqrt(2)) * 
                     np.cos(z/np.sqrt(3)))
        return sum_part - prod_part + 1


def main():
    # Configuration
    gif_path = "gifs/griewank_DE_MEGATESTE.gif"
    np.random.seed(42)
    fig = plt.figure(figsize=(20, 7))
    alpha = 0.3
    NUM_GENERATIONS = 120
    POP_SIZE = 30
    DIMENSIONS = 3
    SPACE = 1000

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
    
    de = DifferentialEvo(
        F=0.4,
        probability_recombination=0.5,
        fitness=lambda ind: griewank(*ind),
        upperBound=np.array([SPACE]*3),
        lowerBound=np.array([-SPACE]*3),
        num_individuals=POP_SIZE,
        num_dimensions=DIMENSIONS
    )

    plot_de(title="Griewank PSO",
            de=de,
            iterations=NUM_GENERATIONS,
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
    