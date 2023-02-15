import numpy as np

# import matplotlib.pyplot as plt

# # For Testing Purposes: Uncomment lines 6-8, 102-132 and 137-145
# num_training_samples = 500
# bias_choice = 0.7
# pt1 = 0.15


def collocation(num_training_samples, bias_choice, pt1):
    """
    Obtaining number of training samples, bias choice and height of trapezium.
    Creating collocation points and returning them
    """
    # For investigating variation only
    # np.random.seed(3333)
    # Defining points on rhs
    xdiv = 0.5  # This should be tested and then kept constant
    pt2 = pt1 * -1

    # Therefore areas of trapezium halves
    A1 = 2 * xdiv * 0.5
    A2 = (pt1 - pt2) * xdiv * 0.5
    A3 = (pt1 - pt2) * (1 - xdiv)
    Top1 = xdiv * (1 - pt1) * 0.5
    Top2 = (1 - pt1) * (1 - xdiv)
    Bot1 = Top1
    Bot2 = Top2
    print(
        f"Biased section is {(A1+A2+A3)*100/2:.1f}% of the total area,\nand contains {bias_choice*100:.1f}% of the points"
    )

    # Number of points in each section
    Biased_n = int(num_training_samples * bias_choice)
    A1_n = int(Biased_n * A1 / (A1 + A2 + A3))
    A2_n = int(Biased_n * A2 / (A1 + A2 + A3))
    A3_n = Biased_n - (A1_n + A2_n)
    Nobias_n = num_training_samples - Biased_n
    Top1_n = int(Nobias_n * Top1 / (Top1 + Top2 + Bot1 + Bot2))
    Top2_n = int(Nobias_n * Top2 / (Top1 + Top2 + Bot1 + Bot2))
    Bot1_n = int(Nobias_n * Bot1 / (Top1 + Top2 + Bot1 + Bot2))
    Bot2_n = Nobias_n - (Top1_n + Top2_n + Bot1_n)

    # Defining triangle vertices
    Top_v = [(0, 1), (xdiv, 1), (xdiv, pt1)]
    Bot_v = [(0, -1), (xdiv, -1), (xdiv, pt2)]
    A1_v = [(0, -1), (0, 1), (xdiv, pt2)]
    A2_v = [(0, 1), (xdiv, pt1), (xdiv, pt2)]

    # Code generating points uniformly on arbitrary triangles blow is from StackOverflow "Mark Dickinson"
    # Link: https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain

    def points_on_triangle(v, n):
        """
        Give n random points uniformly on a triangle.

        The vertices of the triangle are given by the shape
        (2, 3) array *v*: one vertex per row.
        """
        x = np.sort(np.random.rand(2, n), axis=0)
        return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v

    # Using function to create lattice now:
    collocation_top1 = points_on_triangle(Top_v, Top1_n)
    collocation_mid1 = points_on_triangle(A1_v, A1_n)
    collocation_mid2 = points_on_triangle(A2_v, A2_n)
    collocation_bot1 = points_on_triangle(Bot_v, Bot1_n)
    # Rectangle lattices
    collocation_top2 = np.random.rand(Top2_n, 2)
    collocation_mid3 = np.random.rand(A3_n, 2)
    collocation_bot2 = np.random.rand(Bot2_n, 2)
    collocation_top2[..., 1] = (1 - pt1) * collocation_top2[
        ..., 1
    ] + pt1  # x (y axis) from pt1 to 1.0
    collocation_mid3[..., 1] = (pt1 - pt2) * collocation_mid3[
        ..., 1
    ] + pt2  # x from pt2 to pt1
    collocation_bot2[..., 1] = (
        abs(-1 - pt2) * collocation_bot2[..., 1] - 1
    )  # x from -1.0 to pt2
    collocation_top2[..., 0] = (1 - xdiv) * collocation_top2[
        ..., 0
    ] + xdiv  # t (x axis) from xdiv
    collocation_mid3[..., 0] = (1 - xdiv) * collocation_mid3[
        ..., 0
    ] + xdiv  # t from xdiv to 1
    collocation_bot2[..., 0] = (1 - xdiv) * collocation_bot2[
        ..., 0
    ] + xdiv  # t from xdiv to 1
    # And putting it all together:
    collocation = np.concatenate(
        (
            collocation_top1,
            collocation_top2,
            collocation_mid1,
            collocation_mid2,
            collocation_mid3,
            collocation_bot1,
            collocation_bot2,
        ),
        axis=0,
    )
    # # Below is for visualisation
    # fig = plt.figure()
    # plt.xlabel("$t$")
    # plt.ylabel("$x$")
    # plt.ylim([-1, 1])
    # plt.xlim([0, 1])
    # plt.plot(
    #     collocation_bot1[:, 0],
    #     collocation_bot1[:, 1],
    #     "x",
    #     collocation_mid1[:, 0],
    #     collocation_mid1[:, 1],
    #     "x",
    #     collocation_mid2[:, 0],
    #     collocation_mid2[:, 1],
    #     "x",
    #     collocation_top1[:, 0],
    #     collocation_top1[:, 1],
    #     "x",
    #     collocation_bot2[:, 0],
    #     collocation_bot2[:, 1],
    #     "x",
    #     collocation_mid3[:, 0],
    #     collocation_mid3[:, 1],
    #     "x",
    #     collocation_top2[:, 0],
    #     collocation_top2[:, 1],
    #     "x",
    #     markersize=2,
    # )
    # plt.show()

    return collocation


# test_array = collocation(num_training_samples, bias_choice, pt1)
# fig = plt.figure(figsize=[12, 8], dpi=50)
# plt.xlabel("$t$")
# plt.ylabel("$x$")
# plt.ylim([-1, 1])
# plt.xlim([0, 1])
# plt.plot(test_array[:, 0], test_array[:, 1], "x", markersize=2)
# plt.tight_layout()
# plt.show()
