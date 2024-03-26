import autograd.numpy as np
from autograd import grad, jacobian
from numpy.linalg import eig

E_Na = 55.0
g_Na = 40.0
E_K = -77.0
g_K = 35.0
E_L = -65.0
g_L = 0.3
C = 1.0
I_ext = 0.375

def alpha_n(V):
    return 0.02 * (V - 25.0) / (1.0 - np.exp(-(V-25.0)/9.0))

def beta_n(V):
    return -0.002 * (V - 25.0) / (1.0 - np.exp((V-25.0)/9.0))

def alpha_m(V):
    return 0.182 * (V + 35.0) / (1.0 - np.exp(-(V+35.0)/9.0))

def beta_m(V):
    return -0.124 * (V + 35.0) / (1.0 - np.exp((V+35.0)/9.0))

def alpha_h(V):
    return 0.25 * np.exp(-(V+90.0)/12.0)

def beta_h(V):
    return 0.25 * np.exp((V+62.0)/6.0) / np.exp((V+90.0)/12.0)

def eqs(x):

    V, m, n, h = x

    return np.array([
        g_L*(E_L-V) + g_Na*(m**3.0)*h*(E_Na-V) + g_K*(n**4.0)*(E_K-V) + I_ext,
        alpha_m(V)*(1-m) - beta_m(V)*m,
        alpha_n(V)*(1-n) - beta_n(V)*n,
        alpha_h(V)*(1-h) - beta_h(V)*h
    ])

Jacobian = jacobian(eqs)

# Define the point at which to compute the Jacobian
# x = np.array([-60.09034538949499, 0.08286126628477544, 0.0007827628374709542, 0.42109601367748734])
x = np.array([-58.04926217066852, 0.10159214179608508, 0.0009461804432894837, 0.3765203020172575])

# Evaluate the Jacobian matrix at the given point
result = Jacobian(x)

print("Jacobien:")
print(result)

eigenvalues = eig(result)[0]

print("Valeurs propres:")
print(eigenvalues)
