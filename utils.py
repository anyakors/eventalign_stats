from itertools import product

import numpy as np 
import pandas as pd

import csv

from sklearn import svm
from sklearn.cluster import DBSCAN
from sklearn import mixture

from scipy import stats
from scipy.stats import norm, laplace
from scipy.stats import kstwobign
from scipy.special import kolmogorov

def find_missing_kmers(df_control, kmer_N = 5):

    b = ['A','T','C','G']
    kmers = [''.join(x) for x in list(product(b, repeat=kmer_N))]

    for kmer in kmers:
        indx = df_control['kmer'] == kmer
        df_control_indx = df_control[indx]
        if len(df_control_indx)==0:
            print('kmer {} was not found in the control table; all the comparisons will be set to 0'.format(kmer))
        else:
            print('kmer {} has {} elements'.format(kmer, len(df_control_indx)))

    return


def import_to_df(filename, sample_name):

    table = {}
    table['kmer'], table['pos'], table['logdwell'], table['mean'], table['median'] = [], [], [], [], []

    with open(filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        u = 0
        for line in tsvreader:
            if len(line) == 9 and line[0] != 'ref_pos':
                #table['kmer'].append(line[1][1:4])
                table['kmer'].append(line[1])
                table['pos'].append(int(line[0]))
                table['logdwell'].append(np.log(float(line[3])))
                table['mean'].append(float(line[6]))
                table['median'].append(float(line[7]))
            elif line[0]=='ref_pos':
                u += 1
        print('Unique reads for {}: {}'.format(sample_name, u))

    return pd.DataFrame.from_dict(table)


def min_p(data):
    ma = data[data != 0]
    i = ma.min()
    return i


def test_KS(df, control, stats_table, col='logdwell'):

    ks = []
    pv_ori = []
    pv = []
    pv1 = []
    i = 0
    L = len(np.sort(df.pos.unique()))
    x = np.linspace(0,L,L)

    for pos in range(0, L):
        indx = df['pos'] == pos
        df_indx = df[indx]
        kmer = df_indx['kmer'].iloc[0]
        indx = control['kmer'] == kmer
        df_control_indx = control[indx]

        if len(df_control_indx)>0:
            df_indx_dwell = df_indx[col]
            df_indx_dwell.reset_index(drop=True, inplace=True)
            df_control_indx_dwell = df_control_indx[col]
            df_control_indx_dwell.reset_index(drop=True, inplace=True)
            ks_results = stats.ks_2samp(df_indx_dwell, df_control_indx_dwell)
            ks.append(ks_results[0])
            pv_ori.append(ks_results[1])
        else:
            ks.append(0)
            pv_ori.append(0)
    
    #i = min_p(pv_ori)/10
    i = 10e-140
    pv = [x + i for x in pv_ori]
    pv1.append(np.log10(pv))
    pv1 = np.transpose(pv1)

    KS = pd.DataFrame(list(zip(ks, pv1)), columns =['Statistic', 'p-Value'])
    KS = KS.explode('p-Value')

    stats_table['KS_'+col] = KS['Statistic']
    stats_table['KS_pval_'+col] = abs(KS['p-Value'])

    print(stats_table.head())
    
    return stats_table


def test_KW(df, control, stats_table, col='logdwell'):

    kw = []
    pv_ori = []
    pv = []
    pv1 = []
    L = len(np.sort(df.pos.unique()))
    x = np.linspace(0,L,L)

    for pos in range(0,L):
        indx = df['pos'] == pos
        df_indx = df[indx]
        kmer = df_indx['kmer'].iloc[0]
        indx = control['kmer'] == kmer
        df_control_indx = control[indx]

        if len(df_control_indx)>0:
            df_indx_dwell = df_indx[col]
            df_indx_dwell.reset_index(drop=True, inplace=True)
            df_control_indx_dwell = df_control_indx[col]
            df_control_indx_dwell.reset_index(drop=True, inplace=True)
            kw_results = stats.kruskal(df_indx_dwell, df_control_indx_dwell)
            kw.append(kw_results[0])
            pv_ori.append(kw_results[1])
        else:
            kw.append(0)
            pv_ori.append(0)

    stats_table['KW_'+col] = kw
    print(stats_table.head())

    return stats_table


def test_DBSCAN(df, stats_table):
    # performs 2D clustering analysis on the sample data; no control needed, outlier % is found
    # updates stats_table with all the stats for this sample

    dbscan = []
    L = len(np.sort(df.pos.unique()))

    for kmer in range(0, L):
        indx = df['pos'] == kmer
        df_indx = df[indx]
        X = np.array([np.array(df_indx['median']), np.array(df_indx['logdwell'])])
        X = X.T
        y = DBSCAN(eps=3, min_samples=0.05*len(X)).fit_predict(X)
        out = np.count_nonzero(y==-1)/len(X)
        dbscan.append(out)

    stats_table['DBSCAN'] = dbscan
    print(stats_table.head())

    return stats_table


def test_SVM(df, stats_table):

    svms = []
    L = len(np.sort(df.pos.unique()))

    for kmer in range(0, L):
        indx = df['pos'] == kmer
        df_indx = df[indx]
        X = np.array([np.array(df_indx['median']), np.array(df_indx['logdwell'])])
        X = X.T
        a = svm.OneClassSVM(nu=0.1, kernel='linear')
        y = a.fit_predict(X)
        out = np.count_nonzero(y==-1)/len(X)
        svms.append(out)

    stats_table['SVM'] = svms
    print(stats_table.head())

    return stats_table


def test_BGMM(df, stats_table):

    bgmm = []
    L = len(np.sort(df.pos.unique()))

    for kmer in range(0, L):
        indx = df['pos'] == kmer
        df_indx = df[indx]
        X = np.array([np.array(df_indx['median']), np.array(df_indx['logdwell'])])
        X = X.T
        y = mixture.BayesianGaussianMixture(n_components=2, covariance_type='full').fit_predict(X)
        out = np.count_nonzero(y==0)/len(X)
        bgmm.append(out)

    stats_table['BGMM'] = bgmm
    print(stats_table.head())

    return stats_table


def norm_stats(stats_table, cols=['KS','KS_pval','KW']):

    for c in cols:
        stats_table[c] = stats_table[c] / np.linalg.norm(stats_table[c])

    return stats_table


def transform_input(stats_table):

    L = len(stats_table)
    inputs = []

    for i in range(2, L-2):

        temp = []
        for c in stats_table.columns:
            if c!='pos':
                temp.append(stats_table[c].iloc[i-2:i+3])

        inputs.append(temp)

    return np.array(inputs)


def add_flanks(y):

    flank = np.array([0.0, 0.0])
    y = np.concatenate((flank, y), axis=None)
    y = np.concatenate((y, flank), axis=None)

    return y


def new_control_3(filename, sample):
    table = {}
    table['sample'], table['kmer'], table['logdwell'], table['mean'], table['median'] = [], [], [], [], []
    with open(filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        u = 0
        for line in tsvreader:
            if len(line) == 9 and line[0] != 'ref_pos':
                table['sample'].append(sample)
                table['kmer'].append(line[1][1:4])
                table['logdwell'].append(np.log(float(line[3])))
                table['mean'].append(float(line[6]))
                table['median'].append(float(line[7]))
            elif line[0]=='ref_pos':
                u += 1
        print('Unique reads for {}: {}'.format(sample, u))
    return pd.DataFrame.from_dict(table)


def new_control_5(filename, sample):
    table = {}
    table['sample'], table['kmer'], table['logdwell'], table['mean'], table['median'] = [], [], [], [], []
    with open(filename) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        u = 0
        for line in tsvreader:
            if len(line) == 9 and line[0] != 'ref_pos':
                table['sample'].append(sample)
                table['kmer'].append(line[1])
                table['logdwell'].append(np.log(float(line[3])))
                table['mean'].append(float(line[6]))
                table['median'].append(float(line[7]))
            elif line[0]=='ref_pos':
                u += 1
        print('Unique reads for {}: {}'.format(sample, u))
    return pd.DataFrame.from_dict(table)