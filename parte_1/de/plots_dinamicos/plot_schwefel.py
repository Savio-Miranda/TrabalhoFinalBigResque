import imageio
import numpy as np
import matplotlib.pyplot as plt

from de import DifferentialEvo
from parte_1.de.utils.obj_plot import func_plot

def schwefel(x, y, z):
    x_sin = x * np.sin(np.sqrt(np.abs(x)))
    y_sin = y * np.sin(np.sqrt(np.abs(y)))
    z_sin = z * np.sin(np.sqrt(np.abs(z)))
    return (418.9829 * 3) - (x_sin + y_sin + z_sin)

def main():
    # Configuration
    gif_path = "gifs/schwefel_DE.gif"
    np.random.seed(42)
    fig = plt.figure(figsize=(16, 6))  # Ajuste no tamanho da figura
    alpha = 0.3
    NUM_GENERATIONS = 80
    POP_SIZE = 40
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
    fmin = schwefel(xmin, ymin, zmin)

    # Criar dois subplots com proporções diferentes
    ax1 = fig.add_subplot(121, projection='3d')  # 60% para o gráfico 3D
    ax2 = fig.add_subplot(122)                   # 40% para o fitness

    de = DifferentialEvo(
        F=0.8,
        probability_recombination=0.5,
        fitness=lambda ind: schwefel(*ind),
        upperBound=np.array([SPACE]*3),
        lowerBound=np.array([-SPACE]*3),
        num_individuals=POP_SIZE,
        num_dimensions=DIMENSIONS
    )
    
    frames = []
    best_fitness_history = []
    generations = list(range(NUM_GENERATIONS))  # Lista de gerações para o eixo X
    
    for generation in range(NUM_GENERATIONS):
        de.optimize(maximize=False)
        best_solution = min(de.pop, key=lambda ind: schwefel(*ind))
        best_value = schwefel(*best_solution)
        best_fitness_history.append(best_value)
        
        # Limpar os subplots
        ax1.clear()
        ax2.clear()
        
        # Gráfico 3D
        ax1 = func_plot("Schwefel", de.pop, xmin, ymin, zmin, fmin, space, (0, 1300), alpha, ax1, schwefel)
        ax1.text2D(
            0.02, 0.95, 
            f"Geração: {generation}\nMelhor:\nX: {best_solution[0]:.1f}\nY: {best_solution[1]:.1f}\nZ: {best_solution[2]:.1f}\nFitness: {best_value:.2f}",
            transform=ax1.transAxes,
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        # Gráfico de Fitness - Versão melhorada
        ax2.plot(generations[:generation+1], best_fitness_history, 'b-', linewidth=2, label='Melhor Fitness')
        ax2.scatter(generation, best_value, color='red', s=50, zorder=5)
        
        # Configurações do eixo Y (use linear se a variação for pequena)
        if max(best_fitness_history) - min(best_fitness_history) > 100:
            ax2.set_yscale('log')
            ax2.set_ylabel('Fitness (log)')
        else:
            ax2.set_yscale('linear')
            ax2.set_ylabel('Fitness')
            
        ax2.set_xlabel('Geração')
        ax2.set_title('Evolução do Fitness')
        ax2.grid(True, which="both", linestyle='--', alpha=0.7)
        ax2.legend()
        ax2.set_xlim([0, NUM_GENERATIONS])
        
        # Melhorar o layout
        plt.tight_layout(pad=3.0)
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
    
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"Animation saved as '{gif_path}'")

if __name__ == "__main__":
    main()