import re
import urllib

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import export_graphviz

import pydot
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer

from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt

from yellowbrick.datasets import load_credit
from yellowbrick.classifier import classification_report as yb_class_report

# nhanes_2013_df = pd.DataFrame()
from tabulate import tabulate


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

"""
hanes_2013_questionnaire_files = ['ACQ_H.txt', 'ALQ_H.txt', 'BPQ_H.txt', 'CBQ_H.txt', 'CDQ_H.txt', 'CFQ_H.txt',
                                  'CKQ_H.txt', 'CSQ_H.txt', 'DBQ_H.txt', 'DEQ_H.txt', 'DIQ_H.txt', 'DLQ_H.txt',
                                  'DPQ_H.txt', 'DUQ_H.txt', 'ECQ_H.txt', 'FSQ_H.txt', 'HEQ_H.txt', 'HIQ_H.txt',
                                  'HOQ_H.txt', 'HSQ_H.txt', 'HUQ_H.txt', 'IMQ_H.txt', 'INQ_H.txt', 'KIQ_U_H.txt',
                                  'MCQ_H.txt', 'OCQ_H.txt', 'OHQ_H.txt', 'OSQ_H.txt', 'PAQ_H.txt', 'PFQ_H.txt',
                                  'PUQMEC_H.txt', 'RHQ_H.txt', 'RXQASA_H.txt', 'RXQ_RX_H.txt',
                                  'SLQ_H.txt',
                                  'SMQFAM_H.txt', 'SMQRTU_H.txt', 'SMQSHS_H.txt', 'SMQ_H.txt', 'SXQ_H.txt', 'VTQ_H.txt',
                                  'WHQMEC_H.txt', 'WHQ_H.txt',
                                  ]
"""

hanes_2013_questionnaire_files = ['ACQ_H.txt', 'ALQ_H.txt', 'BPQ_H.txt', 'CBQ_H.txt', 'CDQ_H.txt', 'CFQ_H.txt',
                                  'CKQ_H.txt', 'CSQ_H.txt', 'DBQ_H.txt', 'DEQ_H.txt', 'DIQ_H.txt', 'DLQ_H.txt',
                                  'DPQ_H.txt', 'DUQ_H.txt', 'ECQ_H.txt', 'FSQ_H.txt', 'HEQ_H.txt', 'HIQ_H.txt',
                                  'HOQ_H.txt', 'HSQ_H.txt', 'HUQ_H.txt', 'IMQ_H.txt', 'INQ_H.txt', 'KIQ_U_H.txt',
                                  'MCQ_H.txt', 'OCQ_H.txt', 'OHQ_H.txt', 'OSQ_H.txt', 'PAQ_H.txt', 'PFQ_H.txt',
                                  'PUQMEC_H.txt', 'RHQ_H.txt', 'RXQASA_H.txt', 'RXQ_RX_H.txt',
                                  'SLQ_H.txt',
                                  'SMQFAM_H.txt', 'SMQRTU_H.txt', 'SMQSHS_H.txt', 'SMQ_H.txt', 'SXQ_H.txt', 'VTQ_H.txt',
                                  'WHQMEC_H.txt', 'WHQ_H.txt',
                                  ]

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
    return df


# nhanes_2013_df = read_and_merge_data(nhanes_2013_df, 'DEMO_H.txt')
# nhanes_2013_df = read_and_merge_data(nhanes_2013_df, 'HDL_H.txt')

def merge_all_data(dir_prefix, files, df):
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


def drop_interpolate(df):
    # Drop all values where eligstat != 1
    # nhanes_2013_df = nhanes_2013_df.drop(nhanes_2013_df[nhanes_2013_df['eligstat'] != 1].index)
    df.drop(df[df['eligstat'] != 1].index, inplace=True)

    # Drop all NA columns first
    # nhanes_2013_df = nhanes_2013_df.dropna(axis=1, how='all')
    df.dropna(axis=1, how='all', inplace=True)

    #TODO - look at whether dropping object-type columns can be avoided
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


def train_model(df, smote=False):
    # Drop predictor dependent columns from training set
    train_cols_to_drop = ['seqn', 'mortstat', 'ucod_leading', 'permth_int', 'permth_exm', 'diabetes',
                          'hyperten']
    X = df.drop(labels=train_cols_to_drop, axis=1)
    y = df['mortstat']

    # Split first - random state was 123
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=456)

    if smote:
        X_smote, y_smote = smote_classify(X_train, y_train)
    else:
        X_smote, y_smote = (X_train, y_train)

    tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=7)
    tree_clf.fit(X_smote, y_smote)

    estimator = tree_clf
    return estimator, X_train, X_test, y_train, y_test


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
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')


def bagging_classifier(X_train, y_train, X_test, y_test):
    bagged_tree = BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5),
                                    n_estimators=20)

    # Fit to the training data
    bagged_tree.fit(X_train, y_train)

    # Training & testing accuracy score
    train_accuracy = bagged_tree.score(X_train, y_train)
    test_accuracy = bagged_tree.score(X_test, y_test)
    print(f"Bagging classifier - train accuracy: {train_accuracy}  test_accuracy: {test_accuracy}")


def random_forest_classifier(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(n_estimators=100, max_depth=5)
    forest.fit(X_train, y_train)

    # Training & testing accuracy score
    train_accuracy = forest.score(X_train, y_train)
    test_accuracy = forest.score(X_test, y_test)
    print(f"Random forest classifier - train accuracy: {train_accuracy}  test_accuracy: {test_accuracy}")
    return forest


def yb_classification_report(note, tree_clf, X_test, y_test):
    print(note)
    visualizer = yb_class_report(tree_clf, X_test, y_test)
    visualizer.show()


def main():
    # global nhanes_2013_df
    hanes_2013_dir = 'nhanes-2013/'
    hanes_2013_questionnaire_dir = 'nhanes-2013-questionnaire/'

    mort_df = read_mortality_data()
    # nhanes_2013_df = merge_all_data(hanes_2013_dir, hanes_2013_files, mort_df)
    nhanes_2013_df = merge_all_data(hanes_2013_questionnaire_dir,  hanes_2013_questionnaire_files, mort_df)

    # print(tabulate(nhanes_2013_df, headers='keys', tablefmt='psql'))
    # print(nhanes_2013_df.dtypes)

    print(f"After merge: {nhanes_2013_df.shape}")

    nhanes_2013_df = drop_interpolate(nhanes_2013_df)

    print(f"After drop: {nhanes_2013_df.shape}")

    tree_clf, X_train, X_test, y_train, y_test = train_model(nhanes_2013_df, smote=True)

    print(f"Len of full data frame: {len(nhanes_2013_df)}")
    print(f"Len of X_train: {len(X_train)} len of X_test: {len(X_test)}")

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

    bagging_classifier(X_train, y_train, X_test, y_test)
    forest = random_forest_classifier(X_train, y_train, X_test, y_test)

    yb_classification_report("Forest classification", forest, X_test, y_test)


main()
