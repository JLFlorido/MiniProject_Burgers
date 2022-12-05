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
from bias_trapezium_main import run_pinns
import io

biases = [0.28]
# Initialise arrays the results will be appended to.
bias_history = []
time_results = []
mean_results = []
stddev_results = []

for bias_factor in biases:
    bias_history, time_results, mean_results, stddev_results = run_pinns(
        bias_factor, bias_history, time_results, mean_results, stddev_results
    )

df = pd.DataFrame(
    {
        "bias_factor": [bias_history],
        "time_results": [time_results],
        "mean_error": [mean_results],
        "standard deviation": [stddev_results],
    }
)
df.to_csv("case0_run2_results.dat")
print("\n All cases saved")
# df.to_csv("case0_{0:02f}bias.dat".format(bias_factor))
#    with io.open("file_{0:02d}.dat".format(bias_factor), encoding='utf-8') as f:
#       f.write(str(bias_factor))

# Export data, perhaps show relevant plots.
