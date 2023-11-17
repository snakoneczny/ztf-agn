import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
                            f1_score


from utils import pretty_print


def print_summary_table(results, data_label=None):
    if isinstance(results, pd.DataFrame):
        df = get_summary_table(results, data_label)
    else:
        df = stack_summary_tables(
            [get_summary_table(results_df, data_label) for data_label, results_df in tqdm(results.items())])

    pd.options.display.float_format = '{:.2f}'.format
    display(df)
    print(df.to_latex(escape=False, na_rep='', float_format='%.2f', index=False))


def stack_summary_tables(tables):
    n_qso = tables[0].loc[0, 'QSO support']
    n_obj = tables[0].loc[0, 'support']
    qso_supp_id = tables[0].columns.get_loc('QSO support')
    supp_id = tables[0].columns.get_loc('support')
    for table in tables:
        table.insert(qso_supp_id + 1, 'QSO support (%)', table['QSO support'] / n_qso * 100)
        table.insert(qso_supp_id + 2, 'QSO recall global', table['QSO recall'] * table['QSO support (%)'] / 100)

        prec = table['QSO precision']
        rec = table['QSO recall global']
        table.insert(qso_supp_id + 3, 'QSO f1 global', 2 * (prec * rec) / (prec + rec))

        table.insert(supp_id + 4, 'support (%)', table['support'] / n_obj * 100)

    return pd.concat(tables, ignore_index=True)


def get_summary_table(results_df, data_label=None):
    df = pd.DataFrame()
    features_labels = get_feature_labels(results_df.columns)
    for i, features_label in enumerate(features_labels):
        y_pred = results_df['y_pred {}'.format(features_label)]
        y_true = results_df['y_true']
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)

        new_row = {
            'data': data_label,
            'features': features_label,
            'QSO precision': report['QSO']['precision'],
            'QSO recall': report['QSO']['recall'],
            'QSO f1': report['QSO']['f1-score'],
            'QSO support': report['QSO']['support'],
            'accuracy': report['accuracy'],
            'support': report['macro avg']['support'],
        }
        for column, value in new_row.items():
            df.loc[i, column] = value

    return df


def make_reports(results_df, classifiers_dict, feature_sets):
    features_labels = get_feature_labels(results_df.columns)
    for features_label in features_labels:
        print('features: {}'.format(features_label))

        # Make single report
        clf = classifiers_dict[features_label]
        results_df['y_pred'] = results_df['y_pred {}'.format(features_label)]

        feature_names = features_label.split(' + ')
        features = np.concatenate([feature_sets[feature_name] for feature_name in feature_names])

        make_report(results_df, clf, features)
        print('--------------------')


def get_feature_labels(columns):
    return [column[7:] for column in columns if len(column) > 6 and column[:6] == 'y_pred']


def make_report(results_df, clf=None, features=None, label=None):
    y_pred_column = 'y_pred {}'.format(label) if label else 'y_pred'

    # Classification metrics and confusion matrix
    y_test = results_df['y_true']
    y_pred = results_df[y_pred_column]
    print(classification_report(y_test, y_pred, digits=4, output_dict=False))
    if clf:
        plot_confusion_matrix(y_test, y_pred, clf.classes_)

    # Plot results as functions of magnitude and redshift
    results_df['classification outcome'] = results_df.apply(get_clf_label, args=(y_pred_column,), axis=1)
    data = results_df.dropna(subset=['classification outcome'])
    data = data.loc[(data['redshift'] < 5) & (data['mag_median'] > 16)]

    for x in ['n_obs', 'mag_median', 'redshift']:
        # Plot accuracy, precision, recall
        plt.figure()
        min, max = results_df[x].min(), results_df[x].max()
        n_bins = 20
        bins = np.logspace(np.log10(min), np.log10(max), n_bins) if x == 'n_obs' else np.linspace(min, max, n_bins)
        mid_points = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
        results_df['bin'] = pd.cut(results_df[x], bins, include_lowest=True)
        groups = results_df.groupby('bin')
        acc = groups.apply(lambda x: accuracy_score(x['y_true'], x[y_pred_column]))
        qso_f1 = groups.apply(lambda x: f1_score(x['y_true'], x[y_pred_column], average=None,
                                                 labels=['GALAXY', 'QSO', 'STAR'], zero_division=0)[1])
        n_qso = groups.apply(lambda x: x.loc[x['y_true'] == 'QSO'].shape[0])

        # Plot only bins with enough number of QSOs
        mask = n_qso >= 10
        mid_points, acc, qso_f1, n_qso = mid_points[mask], acc[mask], qso_f1[mask], n_qso[mask]

        # Transfer n_obj into zero one
        n_qso = (n_qso - n_qso.min()) / n_qso.max()
        # Transfer n_obj into range of scores
        min = np.min([acc.min(), qso_f1.min()])
        max = np.max([acc.max(), qso_f1.max()])
        n_qso = (n_qso * (max - min)) + min

        plt.plot(mid_points, acc, label='3-class accuracy', alpha=0.9)
        plt.plot(mid_points, qso_f1, label='QSO F1', alpha=0.9)
        plt.plot(mid_points, n_qso, label='QSO distribution', color='grey', alpha=0.6)
        plt.xlabel(pretty_print(x))
        if x == 'n_obs':
            plt.xscale('log')
        plt.legend()
        plt.show()

        # Plot histograms of classification outcome
        plt.figure()
        hue_order = ['TP: QSO', 'FN: galaxy', 'FN: star', 'FP: galaxy', 'FP: star']
        sns.histplot(
            data, x=x, hue='classification outcome', element='step', fill=False,
            log_scale=[False, True], hue_order=hue_order,
        )

        xlim_dict = {
            'n_obs': None,
            'mag_median': [16, 22],
            'redshift': [0, 5],
        }
        plt.xlim(xlim_dict[x])
        plt.xlabel(pretty_print(x))
        plt.ylabel('counts per bin')
        plt.show()

    # Feature importance
    if not clf is None and not features is None:
        plot_feature_ranking(clf, features, n_features=15, n_top_offsets=1)


def get_clf_label(row, y_pred_column=None):
    y_pred_column = y_pred_column if y_pred_column else 'y_pred'
    labels = [
        ('TP: QSO', 'QSO', 'QSO'),
        ('FN: galaxy', 'QSO', 'GALAXY'),
        ('FN: star', 'QSO', 'STAR'),
        ('FP: galaxy', 'GALAXY', 'QSO'),
        ('FP: star', 'STAR', 'QSO'),
    ]
    for clf_label, y_true, y_pred in labels:
        if (row['y_true'] == y_true) & (row[y_pred_column] == y_pred):
            return clf_label
    return None


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def plot_feature_ranking(model, features, n_features=15, n_top_offsets=1, title=None):
    importances = model.feature_importances_ * 100

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