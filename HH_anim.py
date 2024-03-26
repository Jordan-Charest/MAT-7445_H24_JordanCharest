import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

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

# Intégrer les équations

t_final = 100
I_ext_range = np.arange(-0.5, 10, 0.05)
initial = [-60, 0, 0, 0]

def solve(I_ext):
    sol = solve_ivp(eqs, (0, t_final), initial, args=(I_ext,))
    return sol


V_min = 1000
V_max = -1000
n_min = 1000
n_max = -1000
solns = []
for I_ext in I_ext_range:

    sol = solve(I_ext)
    V, m, n, h = sol.y
    

    if max(V) > V_max:
        V_max = max(V)
    if min(V) < V_min:
        V_min = min(V)
    if max(max(m), max(n), max(h)) > n_max:
        n_max = max(max(m), max(n), max(h))
    if min(min(m), min(n), min(h)) < n_min:
        n_min = min(min(m), min(n), min(h))
    
    
    solns.append(sol)

V_min = V_min*1.1
V_max = V_max*1.1
n_min = n_min*1.1
n_max = n_max*1.1


fig, axs = plt.subplots(2, 1, figsize=(10, 8))


line_V, = axs[0].plot([], [], label='V', color='blue')
line_m, = axs[1].plot([], [], label='m', color='red')
line_n, = axs[1].plot([], [], label='n', color='green')
line_h, = axs[1].plot([], [], label='h', color='orange')

# V vs t
axs[0].set_title('V as a function of t')
axs[0].set_xlabel('t')
axs[0].set_ylabel('V')
axs[0].legend()

# m, n, h vs t
axs[1].set_title('m, n and h vs. t')
axs[1].set_xlabel('t')
axs[1].set_ylabel('Values')
axs[1].legend()

# Animation
def update(frame):
    sol = solns[frame]
    V_data, m_data, n_data, h_data = sol.y
    line_V.set_data(sol.t, V_data)
    line_m.set_data(sol.t, m_data)
    line_n.set_data(sol.t, n_data)
    line_h.set_data(sol.t, h_data)
    axs[0].set_title(f'V as a function of t (I_ext = {I_ext_range[frame]:.3f})')
    axs[1].set_title(f'm, n and h vs. t (I_ext = {I_ext_range[frame]:.3f})')
    axs[0].set_xlim(0, t_final)
    axs[1].set_xlim(0, t_final)
    axs[0].set_ylim(V_min, V_max)
    axs[1].set_ylim(n_min, n_max)
    return line_V, line_m, line_n, line_h

# Set animation
ani = FuncAnimation(fig, update, frames=len(solns), blit=False, interval=100)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()