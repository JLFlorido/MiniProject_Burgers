import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

num_train_samples = 800
# create training input
tx_eqn = np.random.rand(num_train_samples, 2)  # t_eqn =  0 ~ +1
tx_eqn[..., 1] = 2 * tx_eqn[..., 1] - 1  # x_eqn = -1 ~ +1

# Instead of above random lattice...
collocation_top = np.random.rand(int(num_train_samples / 8), 2)
collocation_mid = np.random.rand(int(num_train_samples * 3 / 4), 2)
collocation_bot = np.random.rand(int(num_train_samples / 8), 2)
collocation_top[..., 1] = 0.6 * collocation_top[..., 1] + 0.4  # x from 0.4 to 1.0
collocation_mid[..., 1] = 0.8 * collocation_mid[..., 1] - 0.4  # x from -0.4 to 0.4
collocation_bot[..., 1] = (0.6 * collocation_bot[..., 1]) - 1  # x from -1.0 to -0.4

# Put them together again
collocation = np.concatenate(
    (collocation_top, collocation_mid, collocation_bot), axis=0
)
print(collocation.shape)

tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1  # x_ini = -1 ~ +1
tx_ini[..., 0] = 0  # t_ini =  0
tx_bnd = np.random.rand(num_train_samples, 2)  # t_bnd =  0 ~ +1
tx_bnd[..., 1] = 2 * np.round(tx_bnd[..., 1]) - 1  # x_bnd = -1 or +1
# create training output
u_eqn = np.zeros((num_train_samples, 1))  # u_eqn = 0
u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])  # u_ini = -sin(pi*x_ini)
u_bnd = np.zeros((num_train_samples, 1))  # u_bnd = 0
# train the model using L-BFGS-B algorithm
x_train = [tx_eqn, tx_ini, tx_bnd]
y_train = [u_eqn, u_ini, u_bnd]

# Plotting time
fig = plt.figure(figsize=(16, 12), dpi=100)
# plt.style.use("dark_background")
gs = GridSpec(3, 1)
plt.subplot(gs[0, :])
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.ylim([-1, 1])
plt.xlim([0, 1])
plt.plot(tx_eqn[:, 0], tx_eqn[:, 1], "x")

plt.subplot(gs[1, :])
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.ylim([-1, 1])
plt.xlim([0, 1])
plt.plot(
    collocation_bot[:, 0],
    collocation_bot[:, 1],
    "x",
    collocation_mid[:, 0],
    collocation_mid[:, 1],
    "x",
    collocation_top[:, 0],
    collocation_top[:, 1],
    "x",
)

plt.subplot(gs[2, :])
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.ylim([-1, 1])
plt.xlim([0, 1])
plt.plot(collocation[:, 0], collocation[:, 1], "x")

plt.show()

# plt.savefig("desktop_pic.png", dpi=500)
