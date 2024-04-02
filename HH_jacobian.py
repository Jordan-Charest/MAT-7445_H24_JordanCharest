import jax.numpy as jnp
import jax
from numpy.linalg import eig
from HH_eqs import *
from scipy.optimize import root
import matplotlib.pyplot as plt

def eqs(y):
    
    V, m, n, h = y

    return [
        CdV_dt(V, m, n, h, I_ext),
        dm_dt(V, m, n, h, I_ext),
        dn_dt(V, m, n, h, I_ext),
        dh_dt(V, m, n, h, I_ext)
    ]

# Nous devons d'abord déterminer les points fixes du système pour différentes valeurs de I_ext

# I_ext_list = [0, 0.37, 0.375, 0.5, 1, 3, 10]
I_ext_list = jnp.arange(0.2, 0.52, 0.005)
x0 = [-60, 0, 0, 0]
args = ()

fixed_points = []
success_list = []

for I in I_ext_list:
    I_ext = I
    sol = root(eqs, x0, args)
    fixed_points.append(sol.x)
    success_list.append(sol.success)

for i in range(len(fixed_points)):
    print(f"Succès: {success_list[i]}")
    print(f"Point fixe pour I_ext = {I_ext_list[i]:.3f}:")
    print(fixed_points[i])

print("\n")

Jacobian = jax.jacobian(eqs)

eigenvals = []

for i in range(len(fixed_points)):

    x = jnp.asarray(fixed_points[i])

    # Evaluate the Jacobian matrix at the given point
    result = Jacobian(x)

    print(f"Jacobien à I_ext = {I_ext_list[i]:.3f}:")
    print(jnp.asarray(result))

    eigenvalues = eig(result)[0]
    eigenvals.append(eigenvalues)

    print("Valeurs propres:")
    print(eigenvalues)
    print("\n")

split_eig = ([], [], [], [])
for i in range(len(eigenvals)):
    for j in range(4):
        split_eig[j].append(eigenvals[i][j])

fig, ax = plt.subplots()

bif = np.argmin(np.abs(np.asarray(split_eig[1])-np.asarray(split_eig[2])))

ax.plot(I_ext_list, split_eig[1], label="Valeur propre 2")
ax.plot(I_ext_list, split_eig[2], label="Valeur propre 3")
ax.plot(I_ext_list, split_eig[3], color="red", label="Valeur propre 4")
ax.hlines(0, 0.2, 0.5, color="grey", linestyles="dashed")
ax.vlines(0.4, -0.2, 0.2, color="grey", linestyles="dotted")
ax.vlines(I_ext_list[bif], -0.2, 0.2, color="grey", linestyles="dotted")
ax.set_title("Partie réelle des valeurs propres 2, 3 et 4 en fonction de I_ext")
ax.legend()

plt.show()
plt.clf()

fig, ax = plt.subplots()

ax.plot(I_ext_list, split_eig[0], color="blue", label="Valeur propre 1")
ax.legend()
ax.set_title("Partie réelle de la valeur propre 1 en fonction de I_ext")

plt.show()