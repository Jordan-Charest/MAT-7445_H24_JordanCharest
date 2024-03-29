import jax.numpy as np
from numpy.linalg import eig

# Constantes
E_Na = 55
g_Na = 40
E_K = -77
g_K = 35
E_L = -65
g_L = 0.3
C = 1

# Équations auxiliaires
def alpha_n(V):
    return 0.02 * (V - 25) / (1 - np.exp(-(V-25)/9))

def beta_n(V):
    return -0.002 * (V - 25) / (1 - np.exp((V-25)/9))

def alpha_m(V):
    return 0.182 * (V + 35) / (1 - np.exp(-(V+35)/9))

def beta_m(V):
    return -0.124 * (V + 35) / (1 - np.exp((V+35)/9))

def alpha_h(V):
    return 0.25 * np.exp(-(V+90)/12)

def beta_h(V):
    return 0.25 * np.exp((V+62)/6) / np.exp((V+90)/12)

# Équations différentielles

def CdV_dt(V, m, n, h, I_ext):
    return g_L*(E_L-V) + g_Na*(m**3)*h*(E_Na-V) + g_K*(n**4)*(E_K-V) + I_ext

def dm_dt(V, m, n, h, I_ext):
    return alpha_m(V)*(1-m) - beta_m(V)*m

def dn_dt(V, m, n, h, I_ext):
    return alpha_n(V)*(1-n) - beta_n(V)*n

def dh_dt(V, m, n, h, I_ext):
    return alpha_h(V)*(1-h) - beta_h(V)*h

def eqs(t, y, I_ext):

    V, m, n, h = y

    return [
        CdV_dt(V, m, n, h, I_ext),
        dm_dt(V, m, n, h, I_ext),
        dn_dt(V, m, n, h, I_ext),
        dh_dt(V, m, n, h, I_ext)
    ]