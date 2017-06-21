import matplotlib
import matplotlib.pyplot as plt




import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
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
    #plt.show()

    sample_data.to_csv("life_satisfaction_vs_gdp_per_captiva.csv")
    print(sample_data.loc[position_text.keys()])


    sample_data.plot(kind='scatter', x="GDP per Capita", y='Life satisfaction', figsize=(5,3))
    plt.axis([0, 60000, 0, 10])
    X=np.linspace(0, 60000, 1000)
    plt.plot(X, 2*X/100000, "r")
    plt.text(40000, 2.7, r"$\theta_0=0$", fontsize=14, color="r")
    plt.text(40000, 1.8, r"$\theta_1=2 \times 10^{-5}$", fontsize=14, color="r")

    plt.plot(X, 8 - 5*X/100000, "g" )
    plt.text(5000, 9.1, r"$\theta_0=8$", fontsize=14, color="g")
    plt.text(5000, 8.2, r"$\theta_1=-5 \times 10^{-5}$", fontsize=14, color="g")

    plt.plot(X, 4 + 5*X/100000, "b")
    plt.text(5000, 3.5, r"$\theta_0=4$", fontsize=14, color="b")
    plt.text(5000, 2.6, r"$\theta_1=5 \times 10^{-5}$", fontsize=14, color="b")

    save_fig('tweaking_model_param_plot')
    #plt.show()

    lin1 = linear_model.LinearRegression()
    kNearest = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
    Xsample = np.c_[sample_data["GDP per Capita"]]
    ysample = np.c_[sample_data["Life satisfaction"]]
    lin1.fit(Xsample, ysample)
    kNearest.fit(Xsample, ysample)
    t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
    print("t0 : %s, t1 : %s" %(t0,t1))

    sample_data.plot(kind='scatter', x="GDP per Capita", y='Life satisfaction', figsize=(5,3))
    plt.axis(([0, 60000, 0, 10]))
    X=np.linspace(0, 60000, 1000)
    plt.plot(X, t0 + t1*X, "b")
    plt.text(5000, 3.1, r"$\theta_0=4.85$", fontsize=14, color="b")
    plt.text(5000, 2.2, r"$\theta_1=4.91 \times 10^{-5}$", fontsize=14, color="b")
    save_fig('best_fit_model_plot')
    #plt.show()

    cyprus_gdp_per_capita = gdp_per_capita.loc["Cyprus"]["GDP per Capita"]
    print(cyprus_gdp_per_capita)
    cyprus_predicted_life_satisfaction = lin1.predict(cyprus_gdp_per_capita)[0][0]
    knearest_cyprus_predicted_life_satisfaction = kNearest.predict(cyprus_gdp_per_capita)[0][0]
    print("predicted lin reg : %f, predicted k-Nearest : %f" % (cyprus_predicted_life_satisfaction, knearest_cyprus_predicted_life_satisfaction))

    sample_data.plot(kind='scatter', x="GDP per Capita", y='Life satisfaction', figsize=(5,3), s=1)
    X=np.linspace(0, 60000, 1000)
    plt.plot(X, t0 + t1*X, "b")
    plt.axis([0, 60000, 0, 10])
    plt.text(5000, 7.5, r"$\theta_0=%f$" % t0, fontsize=14 , color="b")
    plt.text(5000, 6.5, r"$\theta_1=%f$" % t1, fontsize=14, color="b")
    plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita], [0, cyprus_predicted_life_satisfaction], "r--")
    plt.text(25000, 5.0, r"Predicted = %f" % cyprus_predicted_life_satisfaction, fontsize=14, color="b")
    plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")


    save_fig('best_fit_model_plot')
    #plt.show()


    # put the missing data together
    sample_data.plot(kind='scatter', x="GDP per Capita", y='Life satisfaction', figsize=(8,3))
    plt.axis([0, 110000, 0, 10])

    position_text2 = {
        "Brazil": (1000, 9.0),
        "Mexico": (11000, 9.0),
        #"Chile": (25000, 9.0),
        #"Czech Republic": (35000, 9.0),
        #"Norway": (60000, 3),
        "Switzerland": (72000, 3.0),
        #"Luxembourg": (90000, 3.0),
    }

    for country, pos_text in position_text2.items():
        pos_data_x, pos_data_y = missing_data.loc[country]
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text, arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))


    plt.show()





if __name__ == "__main__":
    doit()
    save_fig('ton', False)

