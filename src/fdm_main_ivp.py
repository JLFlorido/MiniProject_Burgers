"""
fdm_main.py
In this python script I attempt to use odeint but I'm confused as to how to use it
to solve the burger's PDE by combining it with FD.
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time

# u is a vector with every point being at a different x
N = 3200  # 400
nt = 3201  # 101  # Time intervals
dx = 2 / N
b = 0.01 / np.pi  # viscosity term

print("Begin Calculation...\n")
start_time = time.time()
# Define Function to pass to odeint
def burgers(u_int, t, b, N):
    u_local = np.zeros([N + 1])
    u_local[1:N] = u_int
    dudt = np.zeros([N - 1])

    for i in range(1, N):
        dudt[i - 1] = b * (u_local[i + 1] - 2 * u_local[i] + u_local[i - 1]) / (
            dx * dx
        ) - u_local[i] * (u_local[i + 1] - u_local[i - 1]) / (
            2 * dx
        )  # Changed and commented out previous version
    return dudt


# Define Initial Conditions
x0 = np.linspace(-1, 1, N + 1)  #
u0 = -np.sin(np.pi * x0)
u0_int = u0[1:N]

# Define Time Interval
t = np.linspace(0, 1, nt)
t_eval = range(0, 1, 6401)
# Where the solution is computed
# sol = odeint(burgers, u0, t, args=(b))
u_int = solve_ivp(burgers, t, u0_int, t_eval=t_eval, args=(b, N))
u_int = np.transpose(u_int)
# Writing BC into array
## u[:, 0] = u0
u = np.zeros([N + 1, nt])
u[0, :] = 0
u[N, :] = 0
u[1:N, :] = u_int
print(u.shape)

# Print time taken for main code to run
print("--- %s seconds ---" % (time.time() - start_time))

# Export Data ----------------------------------------------------------------------- Change name every run
np.savetxt("results/FDM/u_3200.csv", u, delimiter=",")

print("\n --- Data Exported ---")
# Plot graph
fig = plt.figure(figsize=(7, 4))
plt.pcolormesh(t, x0, u, cmap="rainbow")
plt.xlabel("t")
plt.ylabel("x")
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label("u(t,x)")
cbar.mappable.set_clim(-1, 1)
plt.savefig("figures/FDM/FDM_3200x3201.png", dpi=300)
plt.show()

print(u.shape)
