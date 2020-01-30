import re

import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz

import pydot
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer

from imblearn.combine import SMOTETomek

nhanes_2013_df = pd.DataFrame()

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

codebooks_dict = {}
with open('codebooks.json', 'r') as f:
    codebooks_dict = json.load(f)


def read_mortality_data():
    global nhanes_2013_df

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
    nhanes_2013_df = pd.read_fwf(dir + file, names=cols, colspecs=cspecs, dtype=dtype,
                                 na_values=["."])


def read_and_merge_data(file):
    global nhanes_2013_df
    file_prefix_pat = re.compile('([\w]+)')

    hanes_2013_dir = 'nhanes-2013/'
    new_df = pd.read_sas(hanes_2013_dir + file, format='xport')
    # Map the lowering function to all column names
    new_df.columns = map(str.lower, new_df.columns)

    for c in new_df.columns:
        # If any column has "sample" in the description remove it
        if c in codebooks_dict and ("sample" in codebooks_dict[c].lower() or "weights" in codebooks_dict[c].lower()):
            print(f"dropping: {c}")
            new_df.drop(labels=c, axis=1, inplace=True)

    # Prefix all the columns except seqn with the file prefix
    key_text = file_prefix_pat.search(file)
    prefix = key_text.group(1).lower()
    new_df.columns = map(lambda x: x if x == 'seqn' else x + "_" + prefix, new_df.columns)

    print(f"new columns: {new_df.columns}")
    # Convert sequence number to int
    new_df = new_df.astype({'seqn': 'int'})
    nhanes_2013_df = pd.merge(nhanes_2013_df, new_df, how='left',
                              left_on=['seqn'],
                              right_on=['seqn'])
    return


# nhanes_2013_df = read_and_merge_data(nhanes_2013_df, 'DEMO_H.txt')
# nhanes_2013_df = read_and_merge_data(nhanes_2013_df, 'HDL_H.txt')

def merge_all_data():
    debug = False
    count = 0
    start = 60
    end = 90
    if not debug:
        start = 0
        end = len(hanes_2013_files)
    # Skip files that don't have seqn as a key
    skip_files = ['BFRPOL_H.txt', 'PCBPOL_H.txt', 'PSTPOL_H.txt']
    for i in range(start, min(len(hanes_2013_files), end)):
        f = hanes_2013_files[i]
        # for f in hanes_2013_files:
        if f in skip_files:
            continue
        print(f"Merging: {f}")
        read_and_merge_data(f)
        count += 1


def impute_mean():
    """
    Completely nan columns must have been dropped already
    :return:
    """
    global nhanes_2013_df

    nhanes_2013_df.dropna(axis=1, how='all', inplace=True)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean = imp_mean.fit(nhanes_2013_df)

    new_df = pd.DataFrame(imp_mean.transform(nhanes_2013_df))
    new_df.columns = nhanes_2013_df.columns
    new_df.index = nhanes_2013_df.index
    nhanes_2013_df = new_df


def drop_interpolate():
    global nhanes_2013_df
    # Drop all values where eligstat != 1
    # nhanes_2013_df = nhanes_2013_df.drop(nhanes_2013_df[nhanes_2013_df['eligstat'] != 1].index)
    nhanes_2013_df.drop(nhanes_2013_df[nhanes_2013_df['eligstat'] != 1].index, inplace=True)

    # Drop all NA columns first
    # nhanes_2013_df = nhanes_2013_df.dropna(axis=1, how='all')
    nhanes_2013_df.dropna(axis=1, how='all', inplace=True)
    impute_mean()

    ## BAD!!!
    # means = nhanes_2013_df.mean()
    # nhanes_2013_df = nhanes_2013_df.fillna(means, inplace=True)
    # nhanes_2013_df.fillna(0, inplace=True)

    # nhanes_2013_df = nhanes_2013_df.dropna()

    # Set index column
    nhanes_2013_df.set_index('seqn')


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


def train_model():
    smote = False
    # Drop predictor dependent columns from training set
    train_cols_to_drop = ['mortstat', 'ucod_leading', 'permth_int', 'permth_exm', 'diabetes',
                          'hyperten']
    X = nhanes_2013_df.drop(labels=train_cols_to_drop, axis=1)
    y = nhanes_2013_df['mortstat']

    # Split first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    if smote:
        X_smote, y_smote = smote_classify(X_train, y_train)
    else:
        X_smote, y_smote = (X_train, y_train)

    tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
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
                print(f"Missing key: {c}")
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
        print(f"leaf node: {labels}")
    graph.write_png(out_file)


def main():
    read_mortality_data()
    merge_all_data()

    drop_interpolate()

    print(f"Len of full data frame: {len(nhanes_2013_df)}")

    tree_clf, X_train, X_test, y_train, y_test = train_model()
    print(f"Len of X_train: {len(X_train)} len of X_test: {len(X_test)}")

    input("Continue?")

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

main()

