# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals


# Common imports
import numpy as np
import os
import tarfile
from six.moves import urllib
import pandas as pd
import colorsys
import hashlib

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


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ration, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def split_train_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

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
    #plt.show()
    print ("stop program")

    print("\n\n** spitting train and test data **\n")
    train_set, test_set = split_train_set(housing, 0.2)
    print("train : %d, test : %d" % (len(train_set), len(test_set)))
    print("the fifth median income : %f, should be 4.036800" % train_set["median_income"][5])

    housing_with_id = housing.reset_index() # adds an index column
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

if __name__ == "__main__":
    main()


