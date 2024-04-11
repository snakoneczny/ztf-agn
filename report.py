import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score

from utils import pretty_print, pretty_print_features


def print_summary_table(results_dict, labels=None, filters=None, concise=False):
    filters = ['g', 'r'] if not filters else filters
    data_labels = [label[0] for label in labels] if labels else list(results_dict[filters[0]].keys())
    feature_labels = [label[1] for label in labels] if labels else [None] * len(data_labels)
    df = stack_summary_tables([get_summary_table(results_dict, data_labels[i], filters, feature_labels[i]) for i in range(len(data_labels))], filters)

    if concise:
        new_column_order = ['data', 'features']
        for filter in filters:
            new_column_order.extend([
                'QSO support (%) {}-band'.format(filter),
                'QSO f1 {}-band'.format(filter),
                'QSO f1 global {}-band'.format(filter),
                'accuracy {}-band'.format(filter),
            ])
        df = df[new_column_order]

    pd.options.display.float_format = '{:.2f}'.format
    display(df)
    print(df.to_latex(escape=False, na_rep='', float_format='%.2f', index=False))


def stack_summary_tables(tables, filters):
    for filter in filters:
        n_qso = tables[0].loc[0, 'QSO support {}-band'.format(filter)]
        n_obj = tables[0].loc[0, 'support {}-band'.format(filter)]
        qso_supp_id = tables[0].columns.get_loc('QSO support {}-band'.format(filter))
        supp_id = tables[0].columns.get_loc('support {}-band'.format(filter))
        for table in tables:
            table.insert(qso_supp_id + 1, 'QSO support (%) {}-band'.format(filter), table['QSO support {}-band'.format(filter)] / n_qso)
            table.insert(qso_supp_id + 2, 'QSO recall global {}-band'.format(filter), table['QSO recall {}-band'.format(filter)] * table['QSO support (%) {}-band'.format(filter)])

            prec = table['QSO precision {}-band'.format(filter)]
            rec = table['QSO recall global {}-band'.format(filter)]
            table.insert(qso_supp_id + 3, 'QSO f1 global {}-band'.format(filter), 2 * (prec * rec) / (prec + rec))

            table.insert(supp_id + 4, 'support (%) {}-band'.format(filter), table['support {}-band'.format(filter)] / n_obj * 100)

    return pd.concat(tables, ignore_index=True)


def get_summary_table(results_dict, data_label, filters, feature_labels=None):
    summary_df = pd.DataFrame()
    if feature_labels is None:
        feature_labels = get_feature_labels(results_dict[filters[0]][data_label].columns)

    for i, feature_label in enumerate(feature_labels):
        new_row = {
            'data': data_label,
            'features': feature_label,
        }

        for filter in filters:
            results_df = results_dict[filter][data_label]

            y_pred = results_df['y_pred {}'.format(feature_label)]
            y_true = results_df['y_true']
            report = classification_report(y_true, y_pred, digits=4, output_dict=True)

            new_row.update({
                'QSO precision {}-band'.format(filter): report['QSO']['precision'],
                'QSO recall {}-band'.format(filter): report['QSO']['recall'],
                'QSO f1 {}-band'.format(filter): report['QSO']['f1-score'],
                'QSO support {}-band'.format(filter): report['QSO']['support'],
                'accuracy {}-band'.format(filter): report['accuracy'],
                'support {}-band'.format(filter): report['macro avg']['support'],
            })

        for column, value in new_row.items():
            summary_df.loc[i, column] = value

    return summary_df


def make_reports(results_df_dict, feature_importance_dict, filter, data_label, feature_labels=None):
    results_df = results_df_dict[filter][data_label]
    feature_importance_dict = feature_importance_dict[filter][data_label]

    feature_labels = get_feature_labels(results_df.columns) if not feature_labels else feature_labels
    for features_label in feature_labels:
        print('features: {}'.format(features_label))

        # Make single report
        results_df['y_pred'] = results_df['y_pred {}'.format(features_label)]

        feature_importance = feature_importance_dict[features_label] if features_label in feature_importance_dict else None
        make_report(results_df, feature_importance)
        print('--------------------')


def get_feature_labels(columns):
    return [column[7:] for column in columns if len(column) > 6 and column[:6] == 'y_pred']


def make_report(results_df, feature_importances=None, label=None):
    y_pred_column = 'y_pred {}'.format(label) if label else 'y_pred'
    feature_importances = feature_importances[label] if label and feature_importances else feature_importances

    # Classification metrics and confusion matrix
    y_test = results_df['y_true']
    y_pred = results_df[y_pred_column]
    print(classification_report(y_test, y_pred, digits=4, output_dict=False))
    plot_confusion_matrix(y_test, y_pred, ['GALAXY', 'QSO', 'STAR'])

    # Plot results as functions of magnitude, redshift and number of observations
    x_to_run = [
        'mag_median', 'redshift',
        'n_obs', 'n_obs_200',
        'timespan', 'timespan_200',
        'cadence_mean', 'cadence_mean_200',
        'cadence_median', 'cadence_median_200',
        'cadence_std', 'cadence_std_200'
    ]
    for x in x_to_run:
        plot_results_as_function(results_df, x=x, labels=[y_pred_column], with_accuracy=True)

    # Feature importance
    if feature_importances is not None:
        plot_feature_ranking(feature_importances['features'], feature_importances['importances'],
                             n_features=15, n_top_offsets=1)


def plot_results_as_function(results_df, x, labels, with_accuracy=False):
    plt.figure()

    # Make bins and get number of quasars in bins
    min, max = results_df[x].min(), results_df[x].max()
    n_bins = 20
    bins = np.logspace(np.log10(min), np.log10(max), n_bins) if x in ['n_obs', 'n_obs_200'] else np.linspace(min, max, n_bins)
    mid_points = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    results_df['bin'] = pd.cut(results_df[x], bins, include_lowest=True)
    groups = results_df.groupby('bin')
    n_qso = groups.apply(lambda x: x.loc[x['y_true'] == 'QSO'].shape[0])

    # Make mask based on number of quasars
    mask = n_qso >= 10
    mid_points = mid_points[mask]

    min, max = 1, 0
    for y_pred_column in labels:
        acc = groups.apply(lambda x: accuracy_score(x['y_true'], x[y_pred_column]))
        qso_f1 = groups.apply(lambda x: f1_score(x['y_true'], x[y_pred_column], average=None)[1])

        # Plot only bins with enough number of QSOs
        acc, qso_f1 = acc[mask], qso_f1[mask]

        # Update minimum and maxium plotted values
        min = np.min([min, acc.min(), qso_f1.min()]) if with_accuracy else np.min([min, qso_f1.min()])
        max = np.max([max, acc.max(), qso_f1.max()]) if with_accuracy else np.max([max, qso_f1.max()])

        label = pretty_print_features(' '.join(y_pred_column.split(' ')[1:]))
        if with_accuracy:
            plt.plot(mid_points, acc, label='3-class accuracy {}'.format(label), alpha=0.9)
            label = 'QSO F1 {}'.format(label)
        plt.plot(mid_points, qso_f1, label=label, alpha=0.9)

    n_qso = n_qso[mask]
    # Transfer n_obj into zero one
    n_qso = (n_qso - n_qso.min()) / n_qso.max()
    # Transfer n_obj into range of scores
    n_qso = (n_qso * (max - min)) + min
    # Plot a grey line with number of objects kind of in the background
    plt.plot(mid_points, n_qso, '--', label='QSO distribution', color='grey', alpha=0.5)

    plt.xlabel(pretty_print(x))
    if x in ['n_obs', 'n_obs_200']:
        plt.xscale('log')
    legend_loc = {
        'mag_median': 'lower right',
        'redshift': 'lower center',
        'n_obs': 'lower center',
        'cadence_mean': 'lower center',
        'cadence_std': 'upper right',
    }
    loc = legend_loc[x] if x in legend_loc else None
    plt.legend(loc=loc)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def plot_feature_ranking(features, importances, n_features=15, n_top_offsets=1, title=None):
    importances = np.array(importances) * 100

    indices = np.argsort(importances)[::-1]
    if len(features) > n_features:
        indices = indices[:n_features]

    features_sorted = np.array(features)[indices]
    importances_sorted = np.array(importances)[indices]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(range(len(features_sorted)), importances_sorted, align='center')  # , color=get_cubehelix_palette(1)[0])
    ax.set_yticks(range(len(features_sorted)))

    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis()
    ax.set_xlabel('feature importance (%)')

    val_0 = importances_sorted[0]
    for i, value in enumerate(importances_sorted):
        offset = -0.17 * val_0 if i < n_top_offsets else .01 * val_0
        color = 'white' if i < n_top_offsets else 'black'
        ax.text(value + offset, i + .1, '{:.2f}%'.format(value), color=color)

    ax.grid(False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
