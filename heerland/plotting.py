import matplotlib.pyplot as plt
from pathlib import Path

import heerland.main

def plot_observed_surges():

    filename = "observed_surges.jpg"
    filepath = Path("figures") / filename

    observed_surges = heerland.main.get_observed_surges().sort_values("end_date").reset_index()

    observed_surges["name"] = observed_surges["name"].apply(lambda s: s if "Unnamed" not in s else f"Unnamed ({s.replace('Unnamed glacier ', '')})")

    observation_biases = [1936, 1961, 1970, 1990, 2010]

    plt.figure(figsize=(8, 5))
    plt.title(f"{observed_surges.shape[0]} surges in total. {observed_surges.shape[0] - observed_surges.name.unique().shape[0]} repeated surges.")
    plt.barh(observed_surges["name"], observed_surges["end_date"] - observed_surges["start_date"], left=observed_surges["start_date"])

    plt.vlines(observation_biases, *plt.gca().get_ylim(), color="gray", linestyles=":")
    plt.xlim(1870, 2025)
    plt.tight_layout()
    filepath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(filepath, dpi=300)
    plt.show()
