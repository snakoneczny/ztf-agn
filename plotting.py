import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
from astropy import time


FILTER_COLORS = {'g': 'C2', 'r': 'C3', 'i': 'C4'}


def plot_embedding(embedding, labels, sublabels, features, n_obs):
    # Main classes
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels)

    # Sublasses for each main class
    for cls in ['GALAXY', 'QSO', 'STAR']:
        plt.figure()
        idx = np.where(labels == cls)[0]
        sns.scatterplot(x=embedding[idx, 0], y=embedding[idx, 1], hue=sublabels.loc[idx])

    # AGN only
    plt.figure()
    hue = ['AGN' if (labels[i] == 'AGN') else 'other' for i in range(len(labels))]
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=hue)

    # Magnitude
    plt.figure()
    mag_col = 'percentile_50_g' if 'percentile_50_g' in features else 'median'
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=features[mag_col])

    # Number of observations
    plt.figure()
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=n_obs)


def plot_light_curve(lc_dict):
    dates = time.Time(lc_dict['mjd'], format='mjd').datetime
    filter = lc_dict['filter']

    fig, ax = plt.subplots()
    ax.errorbar(dates, lc_dict['mag'], yerr=lc_dict['magerr'], fmt='o', markersize=0.7)  # , color=FILTER_COLORS[filter])

    ax.invert_yaxis()
    ax.set_ylabel('magnitude {}'.format(filter))
    ax.set_xlabel('date')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
