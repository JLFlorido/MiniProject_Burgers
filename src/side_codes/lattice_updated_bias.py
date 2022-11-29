import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Defining points on rhs
pt1 = 0.2
pt2 = pt1 * -2

# Insert number of training samples and bias
num_training_samples = 2500
bias_choice = 0.9  # Fraction of points in middle section

# Therefore areas of trapezium halves
A1 = 2 * 1 * 0.5
A2 = (pt1 - pt2) * 1 * 0.5

# Code generating points uniformly on arbitrary triangles from S.O.
def points_on_triangle(v, n):
    """
    Give n random points uniformly on a triangle.

    The vertices of the triangle are given by the shape
    (2, 3) array *v*: one vertex per row.
    """
    x = np.sort(np.random.rand(2, n), axis=0)
    return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v


print(
    f"Middle section is {(A1+A2)*100/2}% of the total area,\nand contains {bias_choice*100}% of the points"
)

# Number of points in each section
Top_n = int(num_training_samples * ((1 - bias_choice) / 2))
Bot_n = int(num_training_samples * ((1 - bias_choice) / 2))
A1_n = int(round(num_training_samples * bias_choice * A1 / (A1 + A2)))
A2_n = int(
    num_training_samples - Top_n - Bot_n - A1_n
)  # Done this way to ensure num_training_samples is respected

# Defining triangle vertices
Top_v = [(0, 1), (1, 1), (1, pt1)]
Bot_v = [(0, -1), (1, -1), (1, pt2)]
A1_v = [(0, -1), (0, 1), (1, pt2)]
A2_v = [(0, 1), (1, pt1), (1, pt2)]
# Using function to create lattice now:
collocation_top = points_on_triangle(Top_v, Top_n)
collocation_mid1 = points_on_triangle(A1_v, A1_n)
collocation_mid2 = points_on_triangle(A2_v, A2_n)
collocation_bot = points_on_triangle(Bot_v, Bot_n)
# And putting it all together:
collocation = np.concatenate(
    (collocation_top, collocation_mid1, collocation_mid2, collocation_bot), axis=0
)

# Plotting

fig = plt.figure(figsize=(8, 6), dpi=50)

gs = GridSpec(2, 1)
plt.subplot(gs[0, :])
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.ylim([-1, 1])
plt.xlim([0, 1])
plt.plot(
    collocation_bot[:, 0],
    collocation_bot[:, 1],
    "x",
    collocation_mid1[:, 0],
    collocation_mid1[:, 1],
    "x",
    collocation_mid2[:, 0],
    collocation_mid2[:, 1],
    "x",
    collocation_top[:, 0],
    collocation_top[:, 1],
    "x",
    markersize=2,
)

plt.subplot(gs[1, :])
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.ylim([-1, 1])
plt.xlim([0, 1])
plt.plot(collocation[:, 0], collocation[:, 1], "x", markersize=2)
plt.tight_layout()
plt.show()
