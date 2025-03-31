import numpy as np
import matplotlib.pyplot as plt


def func_plot(title: str, pop: np.ndarray, xmin: float, ymin: float, zmin: float, fmin: float, space: dict, z_lim: tuple, alpha: float, ax: plt.Axes, obj):
    # ========== Visualização da Superfície 3D (z variável) ==========
    x = np.linspace(space["x_min"], space["x_max"], space["resolution"])  # Resolução reduzida para performance
    y = np.linspace(space["y_min"], space["y_max"], space["resolution"])
    X, Y = np.meshgrid(x, y)
    
    # Fixa z = zmin (mínimo global) para a superfície de referência
    Z_surface = obj(X, Y, np.full_like(X, zmin))  # Superfície em z ótimo
    
    ax.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=alpha)

    # ========== Plot dos Indivíduos (x, y, z) com Cores Baseadas no Fitness ==========
    fitness = np.array([obj(*ind) for ind in pop])
    ax.scatter(
        pop[:, 0], pop[:, 1], pop[:, 2] - zmin,  # Coordenadas 3D reais
        c=fitness,  # Cores baseadas no fitness
        cmap=plt.cm.plasma,
        s=50,
        edgecolor='black',
        vmin=z_lim[0], vmax=z_lim[1]  # Normalização das cores
    )

    # ========== Mínimo Global 3D ==========
    # PLOT DO PONTO MÍNIMO GLOBAL
    ax.scatter(
        xmin, ymin, fmin,
        color="red", marker="*", s=200,
        edgecolors="black", label="Mínimo Global"
    )

    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlim(space["x_min"], space["x_max"])
    ax.set_ylim(space["y_min"], space["y_max"])
    ax.set_zlim(z_lim[0], z_lim[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax