import numpy as np
import matplotlib.pyplot as plt

from de import DifferentialEvo
from plot_test import plot_de


def schwefel(x, y, z):
    x_sin = x * np.sin(np.sqrt(np.abs(x)))
    y_sin = y * np.sin(np.sqrt(np.abs(y)))
    z_sin = z * np.sin(np.sqrt(np.abs(z)))
    return (418.9829 * 3) - (x_sin + y_sin + z_sin)


def main():
    # Configuration
    gif_path = "gifs/schwefel_DE_MEGATESTE.gif"
    np.random.seed(42)
    fig = plt.figure(figsize=(20, 7))
    alpha = 0.3
    NUM_GENERATIONS = 60
    POP_SIZE = 30
    DIMENSIONS = 3
    SPACE = 500

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
    
    de = DifferentialEvo(
        F=0.8,
        probability_recombination=0.5,
        fitness=lambda ind: schwefel(*ind),
        upperBound=np.array([SPACE]*3),
        lowerBound=np.array([-SPACE]*3),
        num_individuals=POP_SIZE,
        num_dimensions=DIMENSIONS
    )

    plot_de(title="Schwefel PSO",
            de=de,
            iterations=NUM_GENERATIONS,
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
    