from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
from astropy import time

FILTER_COLORS = {'g': 'C2', 'r': 'C3', 'i': 'C4'}


def plot_embedding(data, feature_labels):
    # Get subset of input data for which given embedding exists
    feature_label = '_'.join(feature_labels)
    data_subset = data.dropna(subset=['t-sne_0_{}'.format(feature_label)])

    # Extract the most important things
    x, y = data_subset['t-sne_0_{}'.format(feature_label)], data_subset['t-sne_1_{}'.format(feature_label)]
    labels = data_subset['CLASS']

    # Main classes
    classes = labels.unique()
    if len(classes) > 1:
        sns.scatterplot(x=x, y=y, hue=labels)

    # Magnitude
    plt.figure()
    sns.scatterplot(x=x, y=y, hue=data_subset['median'])

    # Number of observations
    plt.figure()
    sns.scatterplot(x=x, y=y, hue=data_subset['n_obs'])

    # Sublasses for each main class
    for cls in classes:
        plt.figure()
        idx = (labels == cls)
        sns.scatterplot(x=x.loc[idx], y=y.loc[idx], hue=data_subset['SUBCLASS'].loc[idx])
        plt.title(cls)


def plot_light_curve(lc_dict):
    dates = time.Time(lc_dict['mjd'], format='mjd').datetime
    filter = lc_dict['filter']

    fig, ax = plt.subplots()
    ax.errorbar(dates, lc_dict['mag'], yerr=lc_dict['magerr'], fmt='o',
                markersize=0.7)  # , color=FILTER_COLORS[filter])

    ax.invert_yaxis()
    ax.set_ylabel('magnitude {}'.format(filter))
    ax.set_xlabel('date')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
