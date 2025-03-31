import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de Schwefel
def schwefel(x, y):
    return 418.9829 * 2 - (x * np.sin(np.sqrt(abs(x))) + y * np.sin(np.sqrt(abs(y))))

# Definir limites do gráfico
x = np.linspace(-500, 500, 400)
y = np.linspace(-500, 500, 400)
X, Y = np.meshgrid(x, y)
Z = schwefel(X, Y)

# Ponto mínimo global
xmin, ymin = 420.9687, 420.9687
f_min = schwefel(xmin, ymin)

# Criar a figura e o eixo 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotar a superfície
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Adicionar o ponto mínimo global
ax.scatter(xmin, ymin, f_min, color='red', s=100, edgecolors='black', label="Mínimo Global")

# Configurações do gráfico
ax.set_title("Função de Schwefel (3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
fig.colorbar(surf, label="Valor de f(x, y)")

plt.legend()
plt.show()
