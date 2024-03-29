import jax.numpy as jnp
import jax
from autograd import grad, jacobian
from numpy.linalg import eig
from HH_eqs import *
from scipy.optimize import root

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
I_ext_list = jnp.arange(0.2, 0.46, 0.02)
x0 = [-50, 0, 0, 0]
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
    print(f"Point fixe pour I_ext = {I_ext_list[i]:.2f}:")
    print(fixed_points[i])

print("\n")

Jacobian = jax.jacobian(eqs)

for i in range(len(fixed_points)):

    x = jnp.asarray(fixed_points[i])

    # Evaluate the Jacobian matrix at the given point
    result = Jacobian(x)

    print(f"Jacobien à I_ext = {I_ext_list[i]:.2f}:")
    print(jnp.asarray(result))

    eigenvalues = eig(result)[0]

    print("Valeurs propres:")
    print(eigenvalues)
    print("\n")
