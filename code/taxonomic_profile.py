'''：
prepare taxonomic profile for building classification model
'''
import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, Normalizer, MaxAbsScaler
from tensorflow.keras.utils import to_categorical


def process_ds(data_path, mapping_file, map_stat=1):
    # load data
    df_1 = pd.read_csv(data_path, low_memory=False)
    print(f'load dataset done')
    # extract feature's taxonomy information row to a dataframe
    var_1 = len(df_1) - 1  # row taxonomy, 7560
    var_2 = len(df_1.columns) - 1  # number of columns, 117728
    dfTax = pd.DataFrame(df_1.iloc[var_1, 0:var_2])
    dfTax.reset_index(inplace=True)
    dfTax.rename(columns=dfTax.iloc[0, :], inplace=True)
    dfTax.drop([0], inplace=True)
    dfTax.reset_index(drop=True, inplace=True)
    # split each record by ;
    tax_seq = list(map(lambda x: x.split(';'), list(dfTax['taxonomy'])))  # split each record by ;
    '''
    map=1, process taxonomy information from start.
    map=0, taxonomy information already meet the requirement for assigning embedding vector.
    '''
    if map_stat == 1:
        # load mapping file (ncbi name to gtdb name)
        map_file = pd.read_csv(mapping_file) # mapping_file
        map_dict = dict(zip(map_file['new_ncbi'], map_file['new_gtdb']))
        # reshape each record size, let everyone has 8 element
        tax_dict = {'sk': 0, 'k': 1, 'p': 2, 'c': 3, 'o': 4, 'f': 5, 'g': 6, 's': 7}
        def fun4rank(x):
            tmp_rec = []
            for ele in x:
                tmp_rec.append(ele.split('__'))
            new_rec = ['d__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
            for idx in range(len(tmp_rec)):
                try:
                    k = tmp_rec[idx][0].replace(' ', '')
                    pos = tax_dict[k]
                    if tmp_rec[idx][1] != '':
                        value_ = str(new_rec[pos])+str(tmp_rec[idx][1])
                        if value_ in map_dict.keys():
                            new_rec[pos] = map_dict[value_]
                        else:
                            new_rec[pos] = value_
                except:
                    pass
            return new_rec
        tax_seq_0 = list(map(lambda x: fun4rank(x), tax_seq))
        # change some taxonomy record in mgnify as the same format as gtdb
        d_k = {}
        d_k['d__Archaea'] = 'k__Archaebacteria'
        d_k['d__Bacteria'] = 'k__Eubacteria'
        def fun4extra(x):
            '''
            for keep the same format as gtdb, set as d__Bacteria->'k__Eubacteria'
            'd__Archaea' -> 'k__Archaebacteria'
            'd__Mitochondria' -> 'd__Eukaryota'
            'd__Chloroplast' -> 'd__Eukaryota'
            '''
            if x[0] == 'd__Bacteria' or x[0] == 'd__Archaea':
                if len(x[1]) != 3 and x[1] not in ['k__Bacteria', 'k__Archaea']:
                    print(x[1])
                else:
                    x[1] = d_k[x[0]]
            elif x[1] == 'k__Bacteria':
                if x[0] == 'd__':
                    x[0] = 'd__Bacteria'
                    x[1] = d_k[x[0]]
                else:
                    print(x[0])
            elif x[1] == 'k__Archaea':
                if x[0] == 'd__':
                    x[0] = 'd__Archaea'
                    x[1] = d_k[x[0]]
                else:
                    print(x[0])
            elif x[0] == 'd__Mitochondria' or x[0] == 'd__Chloroplast':
                x[0] = 'd__Eukaryota'
            return x
        tax_seq_1 = list(map(lambda x: fun4extra(x), tax_seq_0))
        dfTax['new_taxonomy'] = tax_seq_1
    else:
        dfTax['new_taxonomy'] = tax_seq
    return df_1, dfTax, var_1, var_2

#main_df, tax_df, row_num, col_num = process_ds()
def text2num(df4tax, emb_path):
    model_n2v = Word2Vec.load(emb_path)# emb_path
    print(f'load trained language model done')
    # assigned embedding to bacteria
    # concat method
    failed_idx = []
    res = []
    for i in range(len(df4tax)):
        item = df4tax['new_taxonomy'][i]
        lyst = []
        for j in range(len(item)):
            try:
                tmp = list(model_n2v.wv[item[j]])
            except:
                tmp = [0] * 10
            lyst = lyst + tmp
        if len(lyst) != 80:
            failed_idx.append(i)
        res.append(lyst)
    print(f'taxonomy embedding done')
    return res

def generate_ds(emb_list, df4tax, ini_df, row_n, col_n, cluster_num=10000):
    matrix_res = np.matrix(emb_list)
    print (f'start building clustering model')
    # AGNES
    agnes_clf = AgglomerativeClustering(n_clusters=cluster_num)
    agnes_clf.fit(matrix_res)
    agnes_hat = agnes_clf.labels_
    df4tax['AGNES'] = agnes_hat
    var = 'AGNES'
    dict_ = {}
    for i in range(len(df4tax)):
        if df4tax[var][i] not in dict_.keys():
            dict_[df4tax[var][i]] = [df4tax['ID'][i]]
        else:
            dict_[df4tax[var][i]].append(df4tax['ID'][i])
    print(f'assigned each feature to calculated group according to {var} method done')
    # merge dataframe by group
    sub_df = ini_df[['ID', 'biom']]
    sub_df.drop(index=row_n, inplace=True)
    cor_df = ini_df.iloc[0:row_n, 1:col_n]
    failed_rec = []
    failed_group = []
    for k in dict_.keys():
        feature_set = dict_[k]
        tmp_col = np.zeros(row_n, )
        for idx in feature_set:
            try:
                tmp_col = tmp_col + cor_df[idx].astype('float')
            except:
                failed_rec.append(idx)
                pass
        try:
            sub_df[f'goup_{k}'] = tmp_col
        except:
            failed_group.append(k)
            pass
    print(f'{var} cluster done')
    return sub_df

#dr_df = generate_ds(emb_matrix, cluster_num=num, tax_df, main_df, row_num, col_num)
def model_input(df):
    X = df.drop(columns=['ID', 'biom'])
    Y = df['biom']
    # convert integers to dummy variables
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    uniqNum_Y = list(set(encoded_Y))
    newY = to_categorical(encoded_Y)
    str_Y = encoder.inverse_transform(uniqNum_Y)
    label_dict = dict(zip(uniqNum_Y, str_Y))
    x_train, x_test, y_train, y_test = train_test_split(X, newY, test_size=0.2, stratify=newY, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    # scale data
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    # reshape dataset
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_train = x_train.astype('float64')
    x_val = x_val.astype('float64')
    x_test = x_test.astype('float64')
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    y_test = y_test.astype('int')
    return x_train, y_train, x_val, y_val, x_test, y_test, label_dict





