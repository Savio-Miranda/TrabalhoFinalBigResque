# plot.py
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from de import DifferentialEvo

def plot_de(title: str, de: DifferentialEvo, iterations: int, xmin: float, ymin: float, zmin: float, 
            fmin: float, space: dict, z_lim: tuple, alpha: float, obj, gif_path):    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121, projection='3d')  # 3D visualization
    ax2 = fig.add_subplot(122)                  # Fitness plot
    
    # Fitness plot configuration
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Fitness Evolution')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    # Fitness history
    best_fitness_history = []

    # Prepare 3D visualization
    x = np.linspace(space["x_min"], space["x_max"], space["resolution"])
    y = np.linspace(space["y_min"], space["y_max"], space["resolution"])
    X, Y = np.meshgrid(x, y)
    Z_surface = obj(X, Y, np.full_like(X, zmin))  # Surface at optimal z
    surf = ax1.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=alpha, edgecolor='none')

    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Function value')
    
    # Initial graphical elements
    population = de.pop
    current_best = min(population, key=lambda ind: de.fitness(ind))
    best_fitness = de.fitness(current_best)
    
    # Plot all individuals - ensure we pass proper arrays
    scatter_individuals = ax1.scatter(
        population[:, 0],
        population[:, 1], 
        # np.clip(population[:, 2], z_lim[0], z_lim[1]),
        np.clip(population[:, 2], z_lim[0], space["resolution"]),
        c='blue', marker='o', alpha=0.7, label='Individuals'
    )
    
    scatter_gbest = ax1.scatter(
        [current_best[0]], 
        [current_best[1]], 
        #[np.clip(current_best[2], z_lim[0], z_lim[1])],
        [np.clip(current_best[2], z_lim[0], space["resolution"])],
        c='red', marker='*', s=200, label='Best Individual'
    )
    
    scatter_min_global = ax1.scatter(
        [xmin], [ymin], [fmin], 
        c='black', marker='X', s=200, label='Global Minimum'
    )
    
    ax1.legend()
    ax1.set_xlim([space["x_min"], space["x_max"]])
    ax1.set_ylim([space['y_min'], space["y_max"]])
    ax1.set_zlim(z_lim[0], z_lim[1])
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    
    # Initial fitness line
    line, = ax2.plot([], [], 'b-', lw=2, label='Best Fitness')
    current_point, = ax2.plot([], [], 'ro', label='Current')
    ax2.legend()
    ax2.set_xlim(0, iterations)
    
    # Info text
    text = ax1.text2D(0.02, 0.95, title, transform=ax1.transAxes)

    def init():
        line.set_data([], [])
        current_point.set_data([], [])
        text.set_text("")
        return [scatter_individuals, scatter_gbest, line, current_point, text]

    def animate(i):
        de.optimize(maximize=False)
        population = de.pop
        current_best = min(population, key=lambda ind: de.fitness(ind))
        best_fitness = de.fitness(current_best)
        clipped_pos_z = np.clip(population[:, 2], z_lim[0], space['resolution'])
        clipped_gbest_z = np.clip(current_best[2], z_lim[0], space['resolution'])
        
        # Update 3D visualization - ensure we pass proper arrays
        scatter_individuals._offsets3d = (
            population[:, 0], 
            population[:, 1], 
            # np.clip(population[:, 2], z_lim[0], z_lim[1])
            clipped_pos_z - zmin,
        )
        
        scatter_gbest._offsets3d = (
            [current_best[0]], 
            [current_best[1]], 
            # [np.clip(current_best[2], z_lim[0], z_lim[1])]
            [clipped_gbest_z - zmin]
        )
        
        # Update history
        best_fitness_history.append(best_fitness)
        
        # Update fitness plot
        generations = range(len(best_fitness_history))
        line.set_data(generations, best_fitness_history)
        current_point.set_data([i], [best_fitness])
        
        # Adjust y-axis limits
        current_min = min(best_fitness_history)
        current_max = max(best_fitness_history)
        ax2.set_ylim(current_min*0.9, current_max*1.1)
        
        # Update text
        text.set_text(
            f"Generation: {i}\n"
            f"Best Position: [{float(current_best[0]):.2f}, {float(current_best[1]):.2f}, {float(current_best[2]):.2f}]\n"
            f"Best Fitness: {best_fitness:.4f}\n"
            f"Global Minimum: {fmin:.4f}"
        )
        
        return [scatter_individuals, scatter_gbest, line, current_point, text]

    # Create animation
    anim = FuncAnimation(fig, animate, frames=iterations, 
                        init_func=init, interval=300, blit=False)
    
    # Save GIF
    print("Saving animation...")
    try:
        # Try first with default writer
        anim.save(gif_path, writer='pillow', fps=10, dpi=100, 
                 savefig_kwargs={'facecolor': 'white'})
        print(f"Animation saved successfully at: {gif_path}")
    except Exception as e:
        print(f"Error saving with pillow: {e}")
        print("Trying alternative method...")
        
        # Manual frame capture method
        frames = []
        for i in range(iterations):
            animate(i)
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            print(f"Frame {i+1}/{iterations} processed", end='\r')
        
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"\nAnimation saved via imageio at: {gif_path}")

    plt.close()