import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from HH_eqs import *

# Intégrer les équations

t_final = 500
I_ext = 0.375
initial = [-60, 0, 0, 0]
# initial = [-60, 0.09655771, 0.02410834, 0.07001662]

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

sol = solve_ivp(eqs, (0, t_final), initial, args=(I_ext,), **integrator_keywords)

t_data = sol.t
(V_data, m_data, n_data, h_data) = sol.y

# On peut décommenter cette ligne pour observer les dernières valeurs des paramètres, p. ex pour voir s'il y a un pt fixe
# print(f"n_timesteps: {len(V_data)}")
# print(f"Final V: {V_data[-10:]}")
# print(f"Final m: {m_data[-10:]}")
# print(f"Final n: {n_data[-10:]}")
# print(f"Final h: {h_data[-10:]}")


fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# V vs t
axs[0, 0].plot(t_data, V_data, label=f'V (initial: {initial[0]})', color='blue')
axs[0, 0].set_title(f"V en fonction de t, I_ext = {I_ext}")
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('V')
axs[0, 0].legend(loc='lower right')

# m, n, h vs t
axs[1, 0].plot(t_data, m_data, label=f'm (initial: {initial[1]})', color='red')
axs[1, 0].plot(t_data, n_data, label=f'n (initial: {initial[2]})', color='green')
axs[1, 0].plot(t_data, h_data, label=f'h (initial: {initial[3]})', color='orange')
axs[0, 0].set_title(f"m, n et h en fonction de t, I_ext = {I_ext}")
axs[1, 0].set_xlabel('t')
axs[1, 0].set_ylabel('m, n, h')
axs[1, 0].legend()

# V vs h
axs[0, 1].plot(h_data, V_data, label=f'V', color='blue')
axs[0, 1].set_title(f'V en fonction de h, I_ext = {I_ext}')
axs[0, 1].set_xlabel('h')
axs[0, 1].set_ylabel('V')

# m vs h
axs[1, 1].plot(h_data, m_data, label=f'm', color='red')
axs[1, 1].set_title(f'm en fonction de h, I_ext = {I_ext}')
axs[1, 1].set_xlabel('h')
axs[1, 1].set_ylabel('m')

plt.subplots_adjust(hspace=0.3)
# plt.savefig(f"{path}/Iext={I_ext}_long.png")
plt.show()
