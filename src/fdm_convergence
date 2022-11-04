import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from scipy.interpolate import griddata
u_100 = pd.read_csv("results/FDM/u_100.csv", header=None)
u_200 = pd.read_csv("results/FDM/u_200.csv", header=None)
u_400 = pd.read_csv("results/FDM/u_400.csv", header=None)
u_800 = pd.read_csv("results/FDM/u_800.csv", header=None)
u_1600 = pd.read_csv("results/FDM/u_1600.csv", header=None)
u_3200 = pd.read_csv("results/FDM/u_3200.csv", header=None)
u_6400 = pd.read_csv("results/FDM/u_6400.csv", header=None)
# -- -- -- -- -- -- -- -- -- --
# Constrict 800 to 400
u_200to100 = u_200.iloc[::2, :]
u_200to100 = u_200to100.iloc[:, ::2]
u_400to200 = u_400.iloc[::2, :]
u_400to200 = u_400to200.iloc[:, ::2]
u_800to400 = u_800.iloc[::2, :]
u_800to400 = u_800to400.iloc[:, ::2]
u_1600to800 = u_1600.iloc[::2, :]
u_1600to800 = u_1600to800.iloc[:, ::2]
u_3200to1600 = u_3200.iloc[::2, :]
u_3200to1600 = u_3200to1600.iloc[:, ::2]
u_6400to3200 = u_6400.iloc[::2, :]
u_6400to3200 = u_6400to3200.iloc[:, ::2]
# -- -- -- -- -- -- -- -- -- --
# Change to np array for math
u_200to100 = pd.DataFrame.to_numpy(u_200to100)
u_400to200 = pd.DataFrame.to_numpy(u_400to200)
u_800to400 = pd.DataFrame.to_numpy(u_800to400)
u_1600to800 = pd.DataFrame.to_numpy(u_1600to800)
u_3200to1600 = pd.DataFrame.to_numpy(u_3200to1600)
u_6400to3200 = pd.DataFrame.to_numpy(u_6400to3200)
# -- -- -- -- -- -- -- -- -- --
L2err_100 = np.sum(np.sum((u_200to100 - u_100) ** 2)) / (101 * 101)
print("The mean error in u for 100->200 was ", L2err_100)
L2err_200 = np.sum(np.sum((u_400to200 - u_200) ** 2)) / (201 * 201)
print("The mean error in u for 200->400 was ", L2err_200)
L2err_400 = np.sum(np.sum((u_800to400 - u_400) ** 2)) / (401 * 401)
print("The mean error in u for 400->800 was ", L2err_400)
L2err_800 = np.sum(np.sum((u_1600to800 - u_800) ** 2)) / (801 * 801)
print("The mean error in u for 800->1600 was ", L2err_800)
L2err_1600 = np.sum(np.sum((u_3200to1600 - u_1600) ** 2)) / (1601 * 1601)
print("The mean error in u for 1600->3200 was ", L2err_1600)
L2err_3200 = np.sum(np.sum((u_6400to3200 - u_3200) ** 2)) / (3201 * 3201)
print("The mean error in u for 3200->6400 was ", L2err_3200)
# -- -- -- -- -- -- -- -- -- --
# grid_x, grid_t = np.mgrid[-1:1:400j, 0:1:400j]
# u_400to800 = griddata(,u_400, (grid_x, grid_t),method='cubic')
