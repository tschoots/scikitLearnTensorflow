# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals


# Common imports
import numpy as np
import os
import tarfile
from six.moves import urllib
import pandas as pd
import colorsys

# to make this notebook's output stable across runs
np.random.seed(42)


# to plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
POJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
TGZ_FILE = "/housing.tgz"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + TGZ_FILE

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    #tgz_path = os.path.join(housing_path, TGZ_FILE )
    tgz_path = housing_path + TGZ_FILE
    print("path : %s" %tgz_path)
    print("url : %s" % housing_url)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(POJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + "png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def main():
    print ("start program")
    fetch_housing_data()
    housing = load_housing_data()
    print(housing.head())
    print("\n** housing.csv info : **\n")
    print(housing.info())

    print("\n** housing decribe ** :")
    print(housing.describe())
    housing.hist(bins=50, figsize=(20,15))
    plt.show()
    print ("stop program")

if __name__ == "__main__":
    main()


