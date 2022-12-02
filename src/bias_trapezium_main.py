"""
bias_main.py Like main, but creating collocation points via a lattice instead of randomly.
This time collocation points made so they can be biased into trapezoid shape specifically.
Also trying to make it so can be automated and save results.
"""
import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import time
import pandas as pd
from lib.collocation import collocation


def run_pinns(bias_factor, bias_history, time_taken, mean_results, stddev_results):
    """
    Test the physics informed neural network (PINN) model for Burgers' equation
    """
    start_time = time.time()
    # number of training, test samples, bias position (pt1).
    num_train_samples = 2500
    num_test_samples = 6401
    pt1 = 0.2
    # kinematic viscosity
    nu = 0.01 / np.pi

    # build a core network model
    network = Network.build()
    # network.summary()
    # build a PINN model
    pinn = PINN(network, nu).build()

    # create training input, collocation points
    tx_eqn = collocation(num_train_samples, bias_factor, pt1)

    # create training input continued
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
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # predict u(t,x) distribution
    # # In theory, if change third argument of t_flat and x_flat linspace from num_test_samples
    # # can choose grid resolution.
    t_flat = np.linspace(
        0, 1, num_test_samples
    )  # This 3/pi was here in old code and moving from notebook version wasn't changed... Keep as 3/pi for consistency
    x_flat = np.linspace(-1, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    saving_start_time = time.time()
    time_taken.append(time.time() - start_time)

    # ------------------------------------------------------
    # plot u(t,x) distribution as a color-map       # CELL 4
    fig = plt.figure(figsize=(10, 8), dpi=50)
    gs = GridSpec(3, 3)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u, cmap="rainbow")
    plt.xlabel("t")
    plt.ylabel("x")
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label("u(t,x)")
    cbar.mappable.set_clim(-1, 1)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0.05, 0.25, 0.5]  # , 0.75, 0.95]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title("t={}".format(t_cs))
        plt.xlabel("x")
        plt.ylabel("u(t,x)")

        # plot second batch of cross sections
    t_cross_sections = [0.75, 0.95, 1]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[2, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title("t={}".format(t_cs))
        plt.xlabel("x")
        plt.ylabel("u(t,x)")

    plt.tight_layout()
    # CHANGE FOR EVERY CASE
    plt.savefig(
        "figures/Bias Results/Type2bias_{0:.2f}.png".format(bias_factor), dpi=300
    )

    # # CHANGE FOR EVERY CASE
    # np.savetxt(
    #     "results/raw/Uend_2,5k_bias3_run0.csv", u, delimiter=","
    # )  # Save vector for plotting all together in one axis

    # The Comparison to FDM
    u_fdm_all = pd.read_csv("results/FDM/u_6400.csv", header=None)  # Import all u
    u_fdm_end = u_fdm_all.iloc[:, -1]  # Extract last vector
    u_fdm_end = pd.DataFrame.to_numpy(
        u_fdm_end
    )  # Changes from dataframe to numpy array

    u_error = np.abs(u[:, 0] - u_fdm_end)
    u_mean = np.mean(u_error)
    u_std = np.std(u_error)

    mean_results.append(u_mean)
    stddev_results.append(u_std)
    bias_history.append(bias_factor)
    print("Case Done, saving took {0:.2f}s".format(time.time() - saving_start_time))
    return bias_history, time_taken, mean_results, stddev_results
