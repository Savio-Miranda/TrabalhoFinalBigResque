import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from swarm import Swarm



def plot_swarm(title: str, swarm: Swarm, iterations: int, xmin: float, ymin: float, zmin: float, fmin: float, space: dict, z_lim: tuple, alpha: float, obj, gif_path):    
    # Criar figura com dois subplots
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121, projection='3d')  # Visualização 3D
    ax2 = fig.add_subplot(122)                  # Gráfico de fitness
    
    # Configuração do gráfico de fitness
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Melhor Fitness')
    ax2.set_title('Evolução do Fitness')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    # Histórico de fitness
    best_fitness_history = []

    # Preparação da visualização 3D
    x = np.linspace(space["x_min"], space["x_max"], space["resolution"])  # Resolução reduzida para performance
    y = np.linspace(space["y_min"], space["y_max"], space["resolution"])
    X, Y = np.meshgrid(x, y)
    Z_surface = obj(X, Y, np.full_like(X, zmin))  # Superfície em z ótimo
    surf = ax1.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=alpha, edgecolor='none')

    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Valor da função')
    
    # Elementos gráficos iniciais
    pos, vel, best_pos, gbest = swarm.get_swarm_status()
    scatter_particles = ax1.scatter(pos[0], pos[1], np.clip(pos[2], 0, space["resolution"]), 
                                  c='blue', marker='o', alpha=0.7, label='Partículas')
    scatter_gbest = ax1.scatter(gbest[0], gbest[1], np.clip(gbest[2], 0, space["resolution"]), 
                              c='red', marker='*', s=200, label='Melhor Global')
    scatter_min_global = ax1.scatter([xmin], [ymin], [fmin], 
                            c='black', marker='X', s=200, label='Mínimo Global')
    ax1.legend()
    

    ax1.set_xlim([space["x_min"], space["x_max"]])
    ax1.set_ylim([space['y_min'], space["y_max"]])
    ax1.set_zlim(z_lim[0], z_lim[1])
    ax1.set_xlabel('Eixo X')
    ax1.set_ylabel('Eixo Y')
    ax1.set_zlabel('Eixo Z')
    
    # Linha de fitness inicial
    line, = ax2.plot([], [], 'b-', lw=2, label='Melhor Fitness')
    current_point, = ax2.plot([], [], 'ro', label='Atual')
    ax2.legend()
    ax2.set_xlim(0, iterations)
    
    # Texto informativo
    text = ax1.text2D(0.02, 0.95, title, transform=ax1.transAxes)

    def init():
        line.set_data([], [])
        current_point.set_data([], [])
        text.set_text("")
        return [scatter_particles, scatter_gbest, line, current_point, text]

    def animate(i):
        swarm.optimize()
        pos, vel, best_pos, gbest = swarm.get_swarm_status()
        
        # Aplica o limite em Z para todas as posições
        clipped_pos_z = np.clip(pos[2], 0, space['resolution'])
        clipped_gbest_z = np.clip(gbest[2], 0, space['resolution'])
        
        # Atualiza visualização 3D com valores limitados
        scatter_particles._offsets3d = (pos[0], pos[1], clipped_pos_z)
        scatter_gbest._offsets3d = (gbest[0], gbest[1], clipped_gbest_z)
        
        # Atualiza histórico
        current_fitness = float(swarm.best_global_value)
        best_fitness_history.append(current_fitness)
        
        # Atualiza gráfico de fitness
        iterations = range(len(best_fitness_history))
        line.set_data(iterations, best_fitness_history)
        current_point.set_data([i], [current_fitness])
        
        # Ajusta limites do eixo y
        current_min = min(best_fitness_history)
        current_max = max(best_fitness_history)
        ax2.set_ylim(current_min*0.9, current_max*1.1)
        
        # Atualiza texto
        text.set_text(
            f"Iteração: {i}\n"
            f"Melhor Posição: [{float(gbest[0]):.2f}, {float(gbest[1]):.2f}, {float(gbest[2]):.2f}]\n"
            f"Melhor Fitness: {current_fitness:.4f}\n"
            f"Mínimo Global: {fmin:.4f}"
        )
        
        # Atualiza quivers a cada 5 iterações (com Z limitado)
        if i % 5 == 0:
            if hasattr(animate, 'quivers'):
                for q in animate.quivers:
                    q.remove()
            vel_norm = np.linalg.norm(vel, axis=0)
            vel_normalized = vel / (vel_norm + 1e-10)
            animate.quivers = [ax1.quiver(pos[0,j], pos[1,j], np.clip(pos[2,j], 0, space["resolution"]),
                              vel_normalized[0,j], vel_normalized[1,j], vel_normalized[2,j],
                              length=space["x_max"]*0.02, color='blue', alpha=0.5)
                             for j in range(swarm.num_particles)]
        
        artists = [scatter_particles, scatter_gbest, line, current_point, text]
        if hasattr(animate, 'quivers'):
            artists.extend(animate.quivers)
        return artists

    # Inicializa quivers
    animate.quivers = []
    
    # Cria animação
    anim = FuncAnimation(fig, animate, frames=iterations, 
                        init_func=init, interval=300, blit=False)
    
    # Salvar GIF
    print("Salvando animação...")
    try:
        # Tenta primeiro com o writer padrão
        anim.save(gif_path, writer='pillow', fps=10, dpi=100, 
                 savefig_kwargs={'facecolor': 'white'})
        print(f"Animação salva com sucesso em: {gif_path}")
    except Exception as e:
        print(f"Erro ao salvar com pillow: {e}")
        print("Tentando método alternativo...")
        
        # Método manual de captura de frames
        frames = []
        for i in range(iterations):
            animate(i)
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            print(f"Frame {i+1}/{iterations} processado", end='\r')
        
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"\nAnimação salva via imageio em: {gif_path}")

    plt.close()
