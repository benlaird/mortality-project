import re
import urllib
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import export_graphviz

import pydot
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.special import expit

from sklearn.impute import SimpleImputer

from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt

from yellowbrick.datasets import load_credit
from yellowbrick.classifier import classification_report as yb_class_report, ClassificationReport
from yellowbrick.classifier import ConfusionMatrix

# nhanes_2013_df = pd.DataFrame()
from tabulate import tabulate

import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
import seaborn as sn

# _hyper = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2}
_hyper = {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2}

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)

hanes_2013_files = ['ALB_CR_H.txt', 'ALDS_H.txt', 'ALD_H.txt', 'AMDGDS_H.txt', 'AMDGYD_H.txt', 'APOB_H.txt',
                    'BFRPOL_H.txt', 'BIOPRO_H.txt', 'CBC_H.txt', 'CHLMDA_H.txt', 'COT_H.txt', 'CUSEZN_H.txt',
                    'DEET_H.txt', 'DEMO_H.txt', 'EPHPP_H.txt', 'ETHOXS_H.txt', 'ETHOX_H.txt', 'FASTQX_H.txt',
                    'FLDEP_H.txt',
                    'FLDEW_H.txt', 'FOLATE_H.txt', 'FOLFMS_H.txt', 'GHB_H.txt', 'GLU_H.txt', 'HCAAS_H.txt',
                    'HCAA_H.txt', 'HDL_H.txt', 'HEPA_H.txt', 'HEPBD_H.txt', 'HEPB_S_H.txt', 'HEPC_H.txt',
                    'HEPE_H.txt', 'HIV_H.txt', 'HPVP_H.txt', 'HPVSWR_H.txt', 'HSV_H.txt', 'IHGEM_H.txt',
                    'INS_H.txt', 'MMA_H.txt', 'OGTT_H.txt', 'ORHPV_H.txt', 'PAH_H.txt', 'PBCD_H.txt',
                    'PCBPOL_H.txt', 'PERNTS_H.txt', 'PERNT_H.txt', 'PFAS_H.txt', 'PHTHTE_H.txt', 'POOLTF_H.txt',
                    'PSTPOL_H.txt', 'SSFLRT_H.txt', 'SSHEPC_H.txt', 'SSPFAC_H.txt', 'SSPFAS_H.txt', 'SSPFSU_H.txt',
                    'SSPHTE_H.txt', 'SSTOCA_H.txt', 'SSTOXO_H.txt', 'TCHOL_H.txt', 'TGEMA_H.txt', 'TRICH_H.txt',
                    'TRIGLY_H.txt', 'TSNA_H.txt', 'TST_H.txt', 'UASS_H.txt', 'UAS_H.txt', 'UCFLOW_H.txt',
                    'UCOTS_H.txt', 'UCOT_H.txt', 'UCPREG_H.txt', 'UHG_H.txt', 'UIO_H.txt', 'UMS_H.txt', 'UM_H.txt',
                    'UTASS_H.txt', 'UTAS_H.txt', 'UVOCS_H.txt', 'UVOC_H.txt', 'VID_H.txt', 'VITB12_H.txt',
                    'VNAS_H.txt', 'VNA_H.txt', 'VOCWBS_H.txt', 'VOCWB_H.txt', ]

# TODO there seems to be a problem merging this file:  'RXQ_RX_H.txt'  so I removed it
hanes_2013_questionnaire_files = ['ACQ_H.txt', 'ALQ_H.txt', 'BPQ_H.txt', 'CBQ_H.txt', 'CDQ_H.txt', 'CFQ_H.txt',
                                  'CKQ_H.txt', 'CSQ_H.txt', 'DBQ_H.txt', 'DEQ_H.txt', 'DIQ_H.txt', 'DLQ_H.txt',
                                  'DPQ_H.txt', 'DUQ_H.txt', 'ECQ_H.txt', 'FSQ_H.txt', 'HEQ_H.txt', 'HIQ_H.txt',
                                  'HOQ_H.txt', 'HSQ_H.txt', 'HUQ_H.txt', 'IMQ_H.txt', 'INQ_H.txt', 'KIQ_U_H.txt',
                                  'MCQ_H.txt', 'OCQ_H.txt', 'OHQ_H.txt', 'OSQ_H.txt', 'PAQ_H.txt', 'PFQ_H.txt',
                                  'PUQMEC_H.txt', 'RHQ_H.txt', 'RXQASA_H.txt',
                                  'SLQ_H.txt',
                                  'SMQFAM_H.txt', 'SMQRTU_H.txt', 'SMQSHS_H.txt', 'SMQ_H.txt', 'SXQ_H.txt', 'VTQ_H.txt',
                                  'WHQMEC_H.txt', 'WHQ_H.txt'
                                  ]

top_n_features = pd.read_csv('topNFeatures.tsv', sep='\t')
top_n_features = top_n_features.iloc[:, 0]
print(f"top N features: {top_n_features}")

codebooks_dict = {}
with open('combined_codebook.json', 'r') as f:
    codebooks_dict = json.load(f)


def read_mortality_data():
    # from pandas.api.extensions import ExtensionDtype
    dir = "data/"
    file = "NHANES_2013_2014_MORT_2015_PUBLIC.dat"

    """
    PERMTH_INT Number of Person Months of Follow-up from NHANES interview date
    Num 8 0-326 Months Number of person-months of follow-up from NHANES
    interview date. Participants who are assumed alive are
    assigned the number of person months at the end of the
    mortality period, December 31, 2015. Only applicable for
    NHANES III and continuous NHANES (1999-2014)
    . .: Ineligible or under age 18
    PERMTH_EXM Number of Person Months of
    Follow-up from NHANES Mobile
    Examination Center (MEC) date
    Num 8 0-326 Months Number of person-months of follow-up from NHANES
    MEC/exam date. Participants who are assumed alive are
    assigned the number of person months at the end of the
    mortality period, December 31, 2015. Only applicable for
    NHANES III and continuous NHANES (1999-2014) """
    """
    dsn <- read_fwf(file=srvyin,
                    col_types = "ciiiiiiiddii",
                    fwf_cols(publicid = c(1,14),
                             eligstat = c(15,15),
                             mortstat = c(16,16),
                             ucod_leading = c(17,19),
                             diabetes = c(20,20),
                             hyperten = c(21,21),
                             dodqtr = c(22,22),
                             dodyear = c(23,26),
                             wgt_new = c(27,34),
                             sa_wgt_new = c(35,42),
                             permth_int = c(43,45),
                             permth_exm = c(46,48)
                    ),
                    na = "."
    """
    t = [(1, 14), (15, 15), (16, 16), (17, 19), (20, 20), (21, 21), (22, 22), (23, 26), (27, 34), (35, 42),
         (43, 45), (46, 48)]

    # new_t = [np.subtract( t, (1, 0))
    cspecs = [(t1 - 1, t2) for (t1, t2) in t]

    friendly_names = [('RIDAGEYR', 'Age in years'), ('RIDRETH3', 'Race'),
                      ('MIAPROXY', 'Proxy used in MEC Interview'),
                      ('WTINT2YR', 'Interview weight'),
                      ('WTMEC2YR', 'Exam weight'),
                      ('LBDHDD', 'Direct HDL-Cholesterol (mg/dL)'),

                      ]
    cols = ['seqn', 'eligstat', 'mortstat', 'ucod_leading', 'diabetes',
            'hyperten', 'dodqtr', 'dodyear', 'wgt_new', 'sa_wgt_new', 'permth_int', 'permth_exm']
    print(cspecs)
    dtype = {'seqn': int, 'eligstat': int, 'mortstat': float, 'ucod_leading': float,
             'diabetes': float, 'hyperten': float, 'dodqtr': float, 'dodyear': float,
             'wgt_new': float, 'sa_wgt_new': float, 'permth_int': float, 'permth_exm': float}
    df = pd.read_fwf(dir + file, names=cols, colspecs=cspecs, dtype=dtype,
                     na_values=["."])
    return df


def read_and_merge_data(dir_prefix, file, df):
    debug = False
    file_prefix_pat = re.compile('([\w]+)')

    new_df = pd.read_sas(dir_prefix + file, format='xport')
    # Map the lowering function to all column names
    new_df.columns = map(str.lower, new_df.columns)

    for c in new_df.columns:
        # If any column has "sample" in the description remove it
        if c in codebooks_dict and ("sample" in codebooks_dict[c].lower() or "weights" in codebooks_dict[c].lower()):
            if debug:
                print(f"dropping: {c}")
            new_df.drop(labels=c, axis=1, inplace=True)
        # Drop all HPV columns except orggh
        if c in codebooks_dict and c.startswith('orx') and c != "orxgh":
            new_df.drop(labels=c, axis=1, inplace=True)

    # Prefix all the columns except seqn with the file prefix
    key_text = file_prefix_pat.search(file)
    prefix = key_text.group(1).lower()
    new_df.columns = map(lambda x: x if x == 'seqn' else x + "_" + prefix, new_df.columns)

    if debug:
        print(f"new columns: {new_df.columns}")
    # Convert sequence number to int
    new_df = new_df.astype({'seqn': 'int'})
    df = pd.merge(df, new_df, how='left',
                  left_on=['seqn'],
                  right_on=['seqn'])
    print(f"After merging file: {file} shape is: {df.shape}")
    return df


# nhanes_2013_df = read_and_merge_data(nhanes_2013_df, 'DEMO_H.txt')
# nhanes_2013_df = read_and_merge_data(nhanes_2013_df, 'HDL_H.txt')

def merge_all_data(dir_prefix, files, df, dummy_run=False):
    debug = False
    count = 0
    start = 60
    end = 90
    if not debug:
        start = 0
        end = len(files)
    # Skip files that don't have seqn as a key
    skip_files = ['BFRPOL_H.txt', 'PCBPOL_H.txt', 'PSTPOL_H.txt', 'RXQ_DRUG.txt']
    for i in range(start, min(len(files), end)):
        f = files[i]
        # for f in files:
        if f in skip_files:
            continue

        # print(f"Merging: {f}")
        df = read_and_merge_data(dir_prefix, f, df)
        count += 1

    if dummy_run:
        return df.head(1000)
    return df


def impute_mean(df):
    """
    Completely nan columns must have been dropped already
    :return:
    """

    df.dropna(axis=1, how='all')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean = imp_mean.fit(df)

    new_df = pd.DataFrame(imp_mean.transform(df))
    new_df.columns = df.columns
    new_df.index = df.index
    return new_df


def drop_interpolate(df, reduce_feature_set=True):
    # Drop all values where eligstat != 1
    # nhanes_2013_df = nhanes_2013_df.drop(nhanes_2013_df[nhanes_2013_df['eligstat'] != 1].index)
    df.drop(df[df['eligstat'] != 1].index, inplace=True)

    # Drop all NA columns first
    # nhanes_2013_df = nhanes_2013_df.dropna(axis=1, how='all')
    df.dropna(axis=1, how='all', inplace=True)

    df = filter_unhealthy_people(df)

    # Keep only the top N important features plus seqn & mortstat
    if reduce_feature_set:
        df = df[top_n_features.append(pd.Series(['seqn', 'mortstat']))]

    # TODO - look at whether dropping object-type columns can be avoided
    df = df.select_dtypes(exclude=['object'])

    df = impute_mean(df)

    # Set index column
    df.set_index('seqn')
    return df


def smote_classify(X, y):
    # X_class, y_class = make_classification(
    #                                n_samples=10000,
    #                                random_state=10,
    #                                n_classes=2,
    #                                n_informative = 4
    #                                      )
    X_class, y_class = make_classification(

        random_state=10,
        n_classes=2,

    )
    print('Original dataset shape %s' % Counter(y))
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res, y_res


def train_model(df, smote=False, sickness_oversample=False):
    # Drop predictor dependent columns from training set
    train_cols_to_drop = ['ucod_leading', 'permth_int', 'permth_exm', 'diabetes',
                          'hyperten']
    sickness_cols = ['rxduse_rxq_rx_h', 'rxddrug_rxq_rx_h', 'rxddrgid_rxq_rx_h', 'rxqseen_rxq_rx_h', 'rxddays_rxq_rx_h',
                     'rxdrsc1_rxq_rx_h', 'rxdrsc2_rxq_rx_h', 'rxdrsc3_rxq_rx_h', 'rxdrsd1_rxq_rx_h', 'rxdrsd2_rxq_rx_h',
                     'rxdrsd3_rxq_rx_h', 'rxdcount_rxq_rx_h', ]

    if set(train_cols_to_drop).issubset(df.columns):
        X = df.drop(labels=train_cols_to_drop, axis=1)
    else:
        X = df.drop(labels=['mortstat', 'seqn'], axis=1)
    y = df['mortstat']

    # Split first - random state was 123
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=456)

    if smote:
        X_smote, y_smote = smote_classify(X_train, y_train)
    elif sickness_oversample:
        meds_file = 'RXQ_RX_H.txt'
        # Join X_test to end of X_train - i.e. df['mortstat']
        # Then merge in the # of meds file
        # Then resplit
        X_comb = pd.concat([X_train, y_train], axis=1)
        print(f"X_comb shape before oversample: {X_comb.shape}")
        X_comb = read_and_merge_data('nhanes-2013-questionnaire/', 'RXQ_RX_H.txt', X_comb)
        X_smote = X_comb.drop(labels=['mortstat'], axis=1)
        # Drop the cols just added
        X_smote = X_smote.drop(labels=sickness_cols, axis=1)
        y_smote = X_comb['mortstat']
        print(f"X_comb shape after oversample: {X_comb.shape}")
    else:
        X_smote, y_smote = (X_train, y_train)

    return X, y, X_smote, X_test, y_smote, y_test


def prefix_from_full_key(key):
    feat_prefix_pat = re.compile('([a-zA-Z0-9]+)')
    key_text = feat_prefix_pat.search(key)
    prefix = key_text.group(1)
    return prefix


def codebook_desc_from_full_key(key):
    prefix = prefix_from_full_key(key)
    return codebooks_dict[prefix]


def get_feature_names(X_train):
    feat_prefix_pat = re.compile('([a-zA-Z0-9]+)')
    # If True, get friendly names of each column in training set
    eng_feat = True
    feat_names = []
    if eng_feat:
        for c in X_train:

            key_text = feat_prefix_pat.search(c)
            prefix = key_text.group(1)

            if prefix in codebooks_dict:
                feat_names.append(f"{codebooks_dict[prefix][0:20]} ({c})")
            else:
                # print(f"Missing key: {c}")
                feat_names.append(f"({c})")
    else:
        feat_names = X_train.columns

    return feat_names


def create_png(filename, X_train, estimator):
    feat_names = get_feature_names(X_train)

    # Export as dot file
    export_graphviz(estimator, out_file=filename + '.dot',
                    feature_names=feat_names,
                    class_names=['Alive', 'Deceased'],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    (graph,) = pydot.graph_from_dot_file(filename + '.dot')
    graph.write_png(filename + '.png')

    ## Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename='tree.png')


from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter


def print_test_tree(in_file, out_file, X_test, y_test, clf):
    feat_names = get_feature_names(X_test)

    (graph,) = pydot.graph_from_dot_file(in_file)

    # empty all nodes, i.e.set color to white and number of samples to zero
    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('\\n')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '\\n'.join(labels))
            node.set_fillcolor('white')

    samples = y_test
    decision_paths = clf.decision_path(X_test)

    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]
            node.set_fillcolor('green')
            labels = node.get_attributes()['label'].split('\\n')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '\\n'.join(labels))
        # print(f"leaf node: {labels}")
    graph.write_png(out_file)


def plot_feature_importances(model, X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8, 8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')


def bagging_classifier(X_train, y_train, X_test, y_test):
    bagged_tree = BaggingClassifier(
        DecisionTreeClassifier(criterion=_hyper['criterion'], max_depth=_hyper['max_depth']),
        n_estimators=20)

    # Fit to the training data
    bagged_tree.fit(X_train, y_train)

    # Training & testing accuracy score
    train_accuracy = bagged_tree.score(X_train, y_train)
    test_accuracy = bagged_tree.score(X_test, y_test)
    print(f"Bagging classifier - train accuracy: {train_accuracy}  test_accuracy: {test_accuracy}")
    return bagged_tree


def random_forest_classifier(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(n_estimators=100,
                                    criterion=_hyper['criterion'],
                                    max_depth=_hyper['max_depth'],
                                    min_samples_split=_hyper['min_samples_split'])
    forest.fit(X_train, y_train)
    test_pred = forest.predict(X_test)
    # Training & testing accuracy score
    train_accuracy = forest.score(X_train, y_train)
    test_accuracy = forest.score(X_test, y_test)
    print(f"Random forest classifier - train accuracy: {train_accuracy}  test_accuracy: {test_accuracy}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    print(classification_report(y_test, test_pred))

    return forest


def yb_classification_report(note, tree_clf, X_test, y_test):
    print(note)

    visualizer = ClassificationReport(tree_clf)

    visualizer.score(X_test, y_test)
    visualizer.show()

    # visualizer = yb_class_report(tree_clf, X_test, y_test)
    # visualizer.score(X_test, y_test)
    # visualizer.show()


def grid_search(X_train, y_train):
    # Results without smote:  {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5}
    # Results with smote:     {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2}
    # Results with smote & top 50:   {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2}
    clf = DecisionTreeClassifier()

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 5, 10],
        'min_samples_split': [2, 5, 10, 15, 20]
    }

    gs_tree = GridSearchCV(clf, param_grid, cv=3)
    gs_tree.fit(X_train, y_train)

    print(gs_tree.best_params_)


def forest_feature_importance(forest, X_train):
    top_n = 100
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    # for f in range(X_train.shape[1]):
    for f in range(top_n):
        print(f"{f + 1}. feature {indices[f]} {importances[indices[f]]} desc: {codebooks_dict[indices[f]]}")

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()


def get_desc(row):
    return codebook_desc_from_full_key(row.name)


def forest_feature_importance_v2(forest, X_train):
    feature_importances = pd.DataFrame(forest.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    feature_importances['desc'] = feature_importances.apply(get_desc, axis=1)

    topn_features = feature_importances.head(50)
    print(tabulate(topn_features, headers='keys', tablefmt='psql'))
    topn_features.to_csv("topNFeatures.tsv", sep='\t')

    return topn_features


def log_regress(X_train, y_train, X_test, y_test):
    clf = LogisticRegression()

    X_scaled = preprocessing.scale(X_train)
    clf.fit(X_scaled, y_train)

    print(clf.coef_)
    print(clf.intercept_)
    y_pred = clf.predict(X_test)
    c_m = confusion_matrix(y_test, y_pred)
    print(c_m)
    yb_classification_report("Logistic regression", clf, X_test, y_test)
    return

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X_train.ravel(), y_train, color='black', zorder=20)

    loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
    plt.plot(X_test, loss, color='red', linewidth=3)
    plt.show()


def xg_boost(X_train, y_train, X_test, y_test):
    param = {
        'eta': 0.3,
        'max_depth': _hyper['max_depth'],
        'objective': 'multi:softprob',
        'num_class': 2}

    steps = 20  # The number of training iterations
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(param, D_train, steps)

    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
    # yb_classification_report("XG Boost", model, X_test, y_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, (best_preds > 0.5))
    print(cm)
    print(classification_report(y_test, (best_preds > 0.5)))
    return model


def plot_correlation(labels, corr):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # cmap=sn.diverging_palette(20, 220, n=200),
    sns_ax = sn.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        square=True,
        # annot=True,
        xticklabels=[codebooks_dict[prefix_from_full_key(k)][0:20] for k in labels],
        yticklabels=[codebooks_dict[prefix_from_full_key(k)][0:20] for k in labels],
        ax=ax
    )
    sns_ax.set_xticklabels(
        sns_ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontsize=8
    );
    sns_ax.set_yticklabels(
        sns_ax.get_yticklabels(),
        fontsize=8
    );
    plt.show()


def correlation_heatmap(df, topn_features):
    features_to_show = 50
    corr = df[topn_features[0:features_to_show]].corr()

    # Zero out correlations less than 0.75
    corr[corr < 0.75] = 0
    labels = topn_features[0:features_to_show]
    plot_correlation(labels, corr)


def correlation_with_dependent(X, y, file):
    # TODO find variables that contain all of the mortstat=1 values but reduce the sample size
    # One way run a one-level decision tree for each column in X, and chose the variable and cut point
    # with the lowest class==0 impurity -- i.e. all the class==1 nodes are in one leaf
    """
    cor = X.corrwith(y, axis=0)
    print(type(cor))
    cor = cor.rename_axis('correlation')
    cor = cor.sort_values()
    print(cor)
    """
    'class_weight={0: 0.9, 1: 0.1}, '
    dir = "correlations/"
    tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=1,
                                      min_samples_split=2, max_features=len(X.columns),
                                      class_weight={0: 0.1, 1: 0.9}, )
    tree_clf.fit(X, y)

    pred = tree_clf.predict(X)
    # Confusion matrix and classification report
    print("Correlation with dependent tree")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred))

    y_score = tree_clf.score(X, y)
    print('Accuracy: ', y_score)

    micro_precision = precision_score(pred, y, average='micro')
    print('Micro-averaged precision score: {0:0.2f}'.format(
        micro_precision))

    macro_precision = precision_score(pred, y, average='macro')
    print('Macro-averaged precision score: {0:0.2f}'.format(
        macro_precision))

    per_class_precision = precision_score(pred, y, average=None)
    print('Per-class precision score:', per_class_precision)

    path = tree_clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    print(f"Impurities: {path.impurities}")

    create_png(dir + file, X, tree_clf)


def correlation_by_indiv_indep_with_dep(X, y):
    count = 0

    for f in X.columns:
        # first col
        train = pd.DataFrame(X[f])
        print(f"{f} correlation with dependent")
        correlation_with_dependent(train, y, f"corr-{f}")


def filter_unhealthy_people(dt):
    return dt[(dt['pfq020_pfq_h'] == 1) | (dt['pfq030_pfq_h'] == 1) |
              (dt['pfq049_pfq_h'] == 1) |
              (dt['pfq051_pfq_h'] == 1) | (dt['pfq054_pfq_h'] == 1) |
              (dt['pfq057_pfq_h'] == 1) |
              (dt['pfq061a_pfq_h'] == 1) |
              (dt['pfq061b_pfq_h'] == 1) |
              (dt['pfq061c_pfq_h'] == 1) |
              (dt['pfq061d_pfq_h'] == 1) |
              (dt['pfq061e_pfq_h'] == 1) |
              (dt['pfq061f_pfq_h'] == 1) |
              (dt['pfq061g_pfq_h'] == 1) |
              (dt['pfq061h_pfq_h'] == 1) |
              (dt['pfq061i_pfq_h'] == 1) |
              (dt['pfq061j_pfq_h'] == 1) |
              (dt['pfq061k_pfq_h'] == 1) |
              (dt['pfq061l_pfq_h'] == 1) |
              (dt['pfq061m_pfq_h'] == 1) |
              (dt['pfq061n_pfq_h'] == 1) |
              (dt['pfq061o_pfq_h'] == 1) |
              (dt['pfq061p_pfq_h'] == 1) |
              (dt['pfq061q_pfq_h'] == 1) |
              (dt['pfq061r_pfq_h'] == 1) |
              (dt['pfq061s_pfq_h'] == 1) |
              (dt['pfq061t_pfq_h'] == 1) |
              (dt['pfq063a_pfq_h'] == 1) |
              (dt['pfq063b_pfq_h'] == 1) |
              (dt['pfq063c_pfq_h'] == 1) |
              (dt['pfq063d_pfq_h'] == 1) |
              (dt['pfq063e_pfq_h'] == 1) |
              (dt['pfq090_pfq_h'] == 1)
              ]


def just_read():
    hanes_2013_dir = 'nhanes-2013/'
    hanes_2013_questionnaire_dir = 'nhanes-2013-questionnaire/'

    mort_df = read_mortality_data()
    mort_df.set_index('seqn')
    print(f"Mortality data: {mort_df.shape}")
    nhanes_2013_df = merge_all_data(hanes_2013_dir, hanes_2013_files, mort_df)
    nhanes_2013_df = merge_all_data(hanes_2013_questionnaire_dir, hanes_2013_questionnaire_files, nhanes_2013_df)
    # nhanes_2013_df = merge_all_data(hanes_2013_questionnaire_dir, ['PFQ_H.txt'], mort_df)

    # print(tabulate(nhanes_2013_df, headers='keys', tablefmt='psql'))
    # print(nhanes_2013_df.dtypes)

    print(f"After merge: {nhanes_2013_df.shape}")

    nhanes_2013_df = drop_interpolate(nhanes_2013_df, reduce_feature_set=False)

    print(f"After drop: {nhanes_2013_df.shape}")
    return nhanes_2013_df


def single_tree(X_train, X_test, y_train, y_test):
    tree_clf = DecisionTreeClassifier(criterion=_hyper['criterion'], max_depth=_hyper['max_depth'])
    tree_clf.fit(X_train, y_train)

    # Train set predictions
    print("*** Training: ***")
    pred = tree_clf.predict(X_train)
    # Confusion matrix and classification report
    print(confusion_matrix(y_train, pred))
    print(classification_report(y_train, pred))

    create_png("train_tree", X_train, tree_clf)

    # Test set predictions
    print("*** Testing: ***")
    test_pred = tree_clf.predict(X_test)
    # Confusion matrix and classification report
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred))

    print_test_tree("train_tree.dot", "test_tree.png", X_test, y_test, tree_clf)
    print(f"Len of X_train: {len(X_train)} len of X_test: {len(X_test)}")


def plot_poisson():
    from scipy.stats import poisson
    data_poisson = poisson.rvs(mu=3, size=10000)
    X = np.arange(80, 120)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle("Poisson distribution of Sickness", fontsize=16)
    # ax.axis('off')
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.plot(X, poisson.pmf(X, 80), 'r-')
    ax.set_xlabel('Sickness')
    ax.set_ylabel('Number of people')
    plt.savefig("sickness.png")


def main(dummy_run=False):
    # global nhanes_2013_df
    hanes_2013_dir = 'nhanes-2013/'
    hanes_2013_questionnaire_dir = 'nhanes-2013-questionnaire/'

    mort_df = read_mortality_data()

    nhanes_2013_df = merge_all_data(hanes_2013_dir, hanes_2013_files, mort_df, dummy_run)
    nhanes_2013_df = merge_all_data(hanes_2013_questionnaire_dir, hanes_2013_questionnaire_files, nhanes_2013_df,
                                    dummy_run)

    # print(tabulate(nhanes_2013_df, headers='keys', tablefmt='psql'))
    # print(nhanes_2013_df.dtypes)

    print(f"After merge: {nhanes_2013_df.shape}")

    nhanes_2013_df = drop_interpolate(nhanes_2013_df)

    print(f"After drop: {nhanes_2013_df.shape}")

    ActionEnum = Enum('Action', ['feature_importance', 'feature_correlation', 'single_tree',
                                 'xg_boost', 'random_forest', 'logistic', 'grid_search'])
    action = ActionEnum.single_tree

    X, y, X_train, X_test, y_train, y_test = train_model(nhanes_2013_df, smote=True, sickness_oversample=False)

    print(f"Total number of seqn: {len(nhanes_2013_df)}")
    print(f"Len of X_train: {len(X_train)} len of X_test: {len(X_test)}")
    print(f"Test set # of deaths: {sum(y_test)}")

    if action == ActionEnum.feature_correlation:
        correlation_by_indiv_indep_with_dep(X, y)
        # correlation_heatmap(X_train, top_n_features)
    elif action == ActionEnum.single_tree:
        single_tree(X_train, X_test, y_train, y_test)
    elif action == ActionEnum.feature_importance:
        forest = RandomForestClassifier(n_estimators=100,
                                        criterion=_hyper['criterion'],
                                        max_depth=_hyper['max_depth'],
                                        min_samples_split=_hyper['min_samples_split'])
        print("Training the forrest over the whole dataframe, length: {len(X}")
        forest.fit(X, y)
        # For feature importance train & test on the whole frame
        forest_feature_importance_v2(forest, X)
    elif action == ActionEnum.random_forest:
        # bagging = bagging_classifier(X_train, y_train, X_test, y_test)
        for i in range(3):
            forest = random_forest_classifier(X_train, y_train, X_test, y_test)
            yb_classification_report("Forest classification", forest, X_test, y_test)
    elif action == ActionEnum.xg_boost:
        clf = xg_boost(X_train, y_train, X_test, y_test)
        """
        model = clf.XGBClassifier()
        kfold = KFold(n_splits=10, random_state=7)
        dt_cv_score = cross_val_score(clf, X_train, y_train, cv=kfold)
        mean_dt_cv_score = np.mean(dt_cv_score)
        print(f"Mean Cross Validation Score: {mean_dt_cv_score :.2%}")
        """
    elif action == ActionEnum.logistic:
        log_regress(X_train, y_train, X_test, y_test)
    elif action == ActionEnum.grid_search:
        grid_search(X_train, y_train)

main()
# just_read()
