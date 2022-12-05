import lib.tf_silent
import numpy as np
import pandas as pd
from main_v2 import run_pinns

# Create list of biases to run through
biases = [0.28, 0.4, 0.5, 0.6, 0.7]
repeats = 2
biases = np.repeat(biases, repeats)

# Initialise arrays the results will be appended to.
bias_history = []
time_results = []
mean_results = []
stddev_results = []

# Run PINNs for every bias, saving to arrays initialised above.
for bias_factor in biases:
    bias_history, time_results, mean_results, stddev_results = run_pinns(
        bias_factor, bias_history, time_results, mean_results, stddev_results
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
    "results/raw/Case0a_run1.csv", encoding="utf-8", index=False
)  # case0_run1_all
print("\n Case saved")

# In future:
# Could add some visualisation of above.
