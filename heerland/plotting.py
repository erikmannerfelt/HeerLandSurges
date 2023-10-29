import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

import heerland.main

def plot_observed_surges():

    filename = "observed_surges.jpg"
    filepath = Path("figures") / filename

    observed_surges = heerland.main.get_observed_surges().sort_values(["end_date", "name"]).reset_index()

    observed_surges = observed_surges.set_index("name").loc[observed_surges.groupby("name")["end_date"].max().sort_values().index].reset_index()

    # observed_surges["name"] = observed_surges["name"].apply(lambda s: s if "Unnamed" not in s else f"Unnamed ({s.replace('Unnamed glacier ', '')})")

    i = 1

    for name in observed_surges["name"].unique():
        if "unnamed" not in name.lower():
            continue

        observed_surges.loc[observed_surges["name"] == name, "name"] = f"Unnamed {i}"
        i += 1

    # observed_surges["name"] = observed_surges["name"].apply(lambda s: s if "Unnamed" not in s else f"Unnamed ({s.replace('Unnamed glacier ', '')})")


    observation_biases = np.array([1936, 1961, 1970, 1990, 2010])

    zero_year = 2023
    yrs_since_surges = np.clip(observed_surges[["start_date", "end_date"]].mean(axis="columns").sort_values().values, None, zero_year)
    yrs_since_surges = pd.Series(np.ones(yrs_since_surges.size), yrs_since_surges).cumsum()

    surges_last_20_yrs = int(yrs_since_surges.iloc[-1] - yrs_since_surges[(zero_year - 20):].iloc[0])
    
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.plot(yrs_since_surges, color="black")
    plt.scatter(yrs_since_surges.index, yrs_since_surges, color="black")
    plt.xlabel("Year")
    plt.ylabel("Cumulative surge count")
    # plt.xlim(plt.gca().get_xlim()[::-1])
    # plt.text(0.03, 0.03, f"{int(yrs_since_surges[:20].iloc[-1])} surges in the last 20 years", transform=plt.gca().transAxes)
    plt.title(f"{surges_last_20_yrs} surges in the last 20 years")
    ylim = plt.gca().get_ylim()
    plt.vlines(zero_year - 20, *ylim, color="red")
    plt.ylim(ylim)

    plt.subplot(1, 2, 2)
    plt.title(f"{observed_surges.shape[0]} surges in total. {observed_surges.shape[0] - observed_surges.name.unique().shape[0]} repeated surges.")
    plt.barh(observed_surges["name"], observed_surges["end_date"] - observed_surges["start_date"], left=observed_surges["start_date"], color="black")

    plt.vlines(observation_biases, *plt.gca().get_ylim(), color="gray", linestyles=":")
    plt.yticks(fontsize=8)
    plt.xlim(1870, 2025)
    plt.xlabel("Year")
    plt.tight_layout()
    filepath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(filepath, dpi=300)
    plt.show()
