import random

import numpy as np
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


def plot_light_curves(light_curves, stats):
    stats = stats.reset_index(drop=True)

    # Sample
    random.seed(3434)
    idx_sample = random.sample(range(len(light_curves)), 8)

    # Sort by redshift
    redshifts = stats.loc[idx_sample, 'Z']
    idx_sample = [i for _, i in sorted(zip(redshifts, idx_sample))]

    # Plot lighcurves
    size = 16
    fig, axs = plt.subplots(4, 2, figsize=(size, size / 210 * 297))
    axs = axs.flatten()

    for i, (idx, ax) in enumerate(zip(idx_sample, axs)):
        lc = light_curves[idx]
        plot_light_curve(lc, ax)

        # Add statistics
        info = [
            'ZTF id: ' + str(lc['id']),
            'ra: ' + str(lc['ra'][0]),
            'dec: ' + str(lc['dec'][0]),
            'class: {}'.format(stats.loc[idx, 'y_true']),
            'redshift: {:.1f}'.format(stats.loc[idx, 'Z']),
            'galaxy: {:.1f}%'.format(stats.loc[idx, 'y_galaxy ZTF + AstrmClf'] * 100),
            'quasar: {:.1f}%'.format(stats.loc[idx, 'y_qso ZTF + AstrmClf'] * 100),
            'star: {:.1f}%'.format(stats.loc[idx, 'y_star ZTF + AstrmClf'] * 100),
            # 'observations:  {}'.format(np.around(stats.loc[idx, 'n obs'])),
            # 'cadence:  ${:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ days'.format(
            #     stats.loc[idx, 'cadence median'], stats.loc[idx, 'cadence plus sigma'], stats.loc[idx, 'cadence minus sigma']),
        ]
        info = '\n'.join(info)
        text_plot = plt.text(.02, .97, info, ha='left', va='top', transform=ax.transAxes)
        text_plot.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='grey'))

        if i >= len(axs) - 2:
            ax.set_xlabel('date')
        if i % 2 == 0:
            ax.set_ylabel('magnitude')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.show()


def plot_light_curve(lc_dict, ax=None, alpha=None):
    if ax is None:
        _, ax = plt.subplots()

    dates = time.Time(lc_dict['mjd'], format='mjd').datetime

    ax.errorbar(dates, lc_dict['mag'], yerr=lc_dict['magerr'], fmt='o',
                markersize=4.0, mfc='white', alpha=alpha)  # color=FILTER_COLORS[filter]

    ax.invert_yaxis()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
