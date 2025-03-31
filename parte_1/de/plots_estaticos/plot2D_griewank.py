import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de griewank
def griewank(x, y, z):
        sum_part = (x**2 + y**2 + z**2) / 4000
        prod_part = (np.cos(x/np.sqrt(1)) * 
                     np.cos(y/np.sqrt(2)) * 
                     np.cos(z/np.sqrt(3)))
        return sum_part - prod_part + 1

# Definir limites do gráfico
x = np.linspace(-1000, 1000, 100)
y = np.linspace(-1000, 1000, 100)
X, Y = np.meshgrid(x, y)
Z = griewank(X, Y, np.full_like(X, 0))

# Ponto mínimo global
xmin, ymin, zmin = 0, 0, 0
f_min = griewank(xmin, ymin, zmin)

# Criar a figura e o eixo 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotar a superfície
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Adicionar o ponto mínimo global
ax.scatter(xmin, ymin, f_min, color='red', s=100, edgecolors='black', label="Mínimo Global")

# Configurações do gráfico
ax.set_title("Função de griewank (3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
fig.colorbar(surf, label="Valor de f(x, y)")

plt.legend()
plt.show()
