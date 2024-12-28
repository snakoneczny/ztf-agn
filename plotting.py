import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
from astropy import time

FILTER_COLORS = {'g': 'C2', 'r': 'C3', 'i': 'C4'}


def plot_heatmaps(data_dict, filter='g'):
    for to_plot in data_dict:
        df = pd.DataFrame.from_records(to_plot[0][filter], columns=['magnitude', 'number of observations', to_plot[1]])
        df = df.loc[(df['magnitude'] >= 18.50) & (df['magnitude'] <= 21.75)]

        pivot = df.pivot(columns='number of observations', index='magnitude', values=to_plot[1])

        fig, ax = plt.subplots(figsize=(5, 6))

        fmt = '.2f' if to_plot[1] == 'QSO F1 score' else '.0f'
        cmap = 'coolwarm_r' if to_plot[1] == 'QSO F1 score' else 'Blues'
        g = sns.heatmap(pivot, annot=True, ax=ax, cbar=False, cmap=cmap, fmt=fmt)

        x = np.arange(0, 5.1, 0.1)
        y = x + 9
        plt.plot(x, y, 'r--', alpha=0.5)
        
        y_labels = ['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()]
        g.set_yticklabels(y_labels)

        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.title(to_plot[1])
        plt.show()


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
            # 'ZTF id: ' + str(lc['id']),
            'ra: ' + str(lc['ra'][0]),
            'dec: ' + str(lc['dec'][0]),
            'class$_\mathrm{{SDSS}}$: {}'.format(stats.loc[idx, 'y_true']),
            'redshift$_\mathrm{{SDSS}}$: {:.1f}'.format(stats.loc[idx, 'Z']),
            'galaxy$_\mathrm{{ZTF}}$: {:.1f}%'.format(stats.loc[idx, 'y_galaxy ZTF + AstrmClf'] * 100),
            'quasar$_\mathrm{{ZTF}}$: {:.1f}%'.format(stats.loc[idx, 'y_qso ZTF + AstrmClf'] * 100),
            'star$_\mathrm{{ZTF}}$: {:.1f}%'.format(stats.loc[idx, 'y_star ZTF + AstrmClf'] * 100),
            # 'observations:  {}'.format(np.around(stats.loc[idx, 'n obs'])),
            # 'cadence:  ${:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ days'.format(
            #     stats.loc[idx, 'cadence median'], stats.loc[idx, 'cadence plus sigma'], stats.loc[idx, 'cadence minus sigma']),
        ]
        info = '\n'.join(info)
        text_plot = plt.text(.02, .97, info, ha='left', va='top', transform=ax.transAxes)
        text_plot.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='grey'))

        # Add letter indicator
        letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        ax.annotate(
            '({})'.format(letters[i]),
            xy=(0.5, 0.07), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0)
        )
        
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
