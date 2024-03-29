import lib.tf_silent
import numpy as np
import pandas as pd
from main_v2 import run_pinns
import time

# Time to run all biases, to help plan better
autorun_starttime = time.time()
# Create list of biases to run through
biases = [0.362, 0.44, 0.5, 0.6]
repeats = 15
biases = np.repeat(biases, repeats)
# a = [0.32, 0.32, 0.36, 0.36, 0.45, 0.45]
# biases = np.append(biases, a)

# Initialise arrays the results will be appended to.
bias_history = []
time_results = []
mean_results = []
stddev_results = []
counter = 0
# Run PINNs for every bias, saving to arrays initialised above.
for bias_factor in biases:
    bias_history, time_results, mean_results, stddev_results, counter = run_pinns(
        bias_factor, bias_history, time_results, mean_results, stddev_results, counter
    )

# Save results as a dataframe, export to a csv file
df = pd.DataFrame(
    {
        "bias_factor": bias_history,
        "time_results": time_results,
        "mean_error": mean_results,
        "standard deviation": stddev_results,
    }
)
df.to_csv(
    "results/raw/Rate_noseed_3.csv", encoding="utf-8", index=False
)  # case0_run1_all
print("\n All runs ({}) done and results saved".format(len(biases)))

# Time taken:
print(
    "\n Total time taken: {0:.1f} minutes".format(
        (time.time() - autorun_starttime) / 60
    )
)

# In future:
# Could add some visualisation of above.
