import matplotlib.pyplot as plt

import heerland.main

def plot_observed_surges():

    observed_surges = heerland.main.get_observed_surges().sort_values("end_date").reset_index()

    plt.barh(observed_surges["name"], observed_surges["end_date"] - observed_surges["start_date"], left=observed_surges["start_date"])
    plt.xlim(1870, 2025)
    plt.tight_layout()
    plt.show()
