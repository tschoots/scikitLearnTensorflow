import matplotlib
import matplotlib.pyplot as plt




import numpy as np
import pandas as pd
import sklearn
import os


def save_fig(fig_id, tight_layout=True):
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "fundamentals"
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def doit():
    # load the data
    oecd_bli = pd.read_csv("BLI_16062017123934543.csv", thousands=',')


    # Prepare the data
    prr = oecd_bli
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    print(oecd_bli.head(2))

    print(oecd_bli["Life satisfaction"].head(5))

    print(set(prr["INEQUALITY"]))
    print(set(prr["Inequality"]))

    # Load and prepare GDP per capita data
    gdp_per_capita = pd.read_csv("weoreptc.aspx", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")
    gdp_per_capita.rename(columns={"2015": "GDP per Capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    print(gdp_per_capita.head(2))

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per Capita", inplace=True)
    print(full_country_stats)

    print(full_country_stats[["GDP per Capita", 'Life satisfaction']].loc["United States"])

    remove_indices = [0, 1, 6, 8, 33, 36]
    keep_indices = list(set(range(36)) - set(remove_indices))

    sample_data = full_country_stats[["GDP per Capita", 'Life satisfaction']].iloc[keep_indices]
    missing_data = full_country_stats[["GDP per Capita", 'Life satisfaction']].iloc[remove_indices]

    sample_data.plot(kind='scatter', x="GDP per Capita", y='Life satisfaction', figsize=(5,3))
    plt.axis([0, 60000, 0, 10])
    position_text = {
        "Hungary" : (5000, 1),
        "Korea" : (18000, 1.7),
        "France" : (29000, 2.4),
        "Australia": (40000, 3.0),
        "United States": (52000, 3.8),
    }
    for country, pos_text in position_text.items():
        pos_data_x, pos_data_y = sample_data.loc[country]
        country = "U.S." if country == "United States" else country
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                     arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
        plt.plot(pos_data_x, pos_data_y, "ro")
    save_fig('money_happy_scatterplot')
    plt.show()


if __name__ == "__main__":
    doit()
    save_fig('ton', False)

