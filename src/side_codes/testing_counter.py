def test_counter(counter):
    bias_factor = 1.5
    print(
        "figures/Bias Results/Rate_seed1_0_{0:.2f}_{1:d}.png".format(
            bias_factor, int(counter)
        )
    )
    counter = counter + 1
    return counter


counter = 0
for x in range(5):
    counter = test_counter(counter)
