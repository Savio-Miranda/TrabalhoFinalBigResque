import imageio
import numpy as np
import matplotlib.pyplot as plt

from de import DifferentialEvo
from plot import func_plot


def griewank(x, y, z):
        sum_part = (x**2 + y**2 + z**2) / 4000
        prod_part = (np.cos(x/np.sqrt(1)) * 
                     np.cos(y/np.sqrt(2)) * 
                     np.cos(z/np.sqrt(3)))
        return sum_part - prod_part + 1


def main():
    # Configuration
    gif_path = "gifs/griewank_DE.gif"
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

    # Criar dois subplots: esquerda para 3D, direita para fitness
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Configurar o gráfico de fitness
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel('Geração')
    ax2.set_ylabel('Melhor Fitness')
    ax2.set_title('Evolução do Fitness')
    ax2.grid(True)
    
    de = DifferentialEvo(
        F=0.4,
        probability_recombination=0.5,
        fitness=lambda ind: griewank(*ind),
        upperBound=np.array([SPACE]*3),
        lowerBound=np.array([-SPACE]*3),
        num_individuals=POP_SIZE,
        num_dimensions=DIMENSIONS
    )

    frames = []
    best_fitness_history = []  # Para armazenar o histórico de fitness
    
    for generation in range(NUM_GENERATIONS):
        # Otimiza uma geração
        de.optimize(maximize=False)

        best_solution = min(de.pop, key=lambda ind: griewank(*ind))
        best_value = griewank(*best_solution)
        best_fitness_history.append(best_value)
        
        # Limpar e preparar os subplots
        ax1.clear()
        ax2.clear()
        
        # Plotar a função 3D
        ax1 = func_plot("Griewank", de.pop, xmin, ymin, zmin, fmin, space, (0, 400), alpha, ax1, griewank)
        
        # Configurar texto no gráfico 3D
        ax1.text2D(
            0.02, 0.95, 
            f"Geração: {generation}\nMelhor Solução:\nX: {best_solution[0]:.2f}\nY: {best_solution[1]:.2f}\nZ: {best_solution[2]:.2f}\nFitness: {best_value:.2f}",
            transform=ax1.transAxes,
            fontsize=10,
        )
        
        # Plotar a curva de fitness
        ax2.plot(best_fitness_history, 'b-', linewidth=2)
        ax2.set_xlabel('Geração')
        ax2.set_ylabel('Melhor Fitness')
        ax2.set_title('Evolução do Fitness')
        ax2.grid(True)
        ax2.set_xlim([0, NUM_GENERATIONS])
        
        # Adicionar ponto atual
        ax2.scatter([generation], [best_value], color='red', zorder=1)
        
        # Renderizar frame
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
    
    # Save animation
    imageio.mimsave(gif_path, frames, fps=15)
    print("Animation saved as 'griewank_DE.gif'")

if __name__ == "__main__":
    main()
    