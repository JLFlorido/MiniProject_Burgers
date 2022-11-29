import numpy as np


def collocation(num_training_samples, bias_choice, pt1):
    """
    Obtaining number of training samples, bias choice and height of trapezium.
    Creating collocation points and returning them
    """
    # Defining points on rhs
    pt2 = pt1 * -2

    # Therefore areas of trapezium halves
    A1 = 2 * 1 * 0.5
    A2 = (pt1 - pt2) * 1 * 0.5

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
    collocation_top = points_on_triangle(Top_v, Top_n)
    collocation_mid1 = points_on_triangle(A1_v, A1_n)
    collocation_mid2 = points_on_triangle(A2_v, A2_n)
    collocation_bot = points_on_triangle(Bot_v, Bot_n)
    # And putting it all together:
    collocation = np.concatenate(
        (collocation_top, collocation_mid1, collocation_mid2, collocation_bot), axis=0
    )
    return collocation
