'''
prepare functional profile for building classification model
using interpro abundance table to build the classification model
build model on the dataset with description
'''
import os
import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def process_ds(fa_path, tax_path):
    fa_data = pd.read_csv(fa_path,low_memory=False)
    fa_data = fa_data.rename(columns={'interpro_accession':'ID'})
    row_n = len(fa_data)-1
    fa_data.iloc[row_n,0] = 'add'
    df_1 = pd.read_csv(tax_path, low_memory=False)  # shape 7561*117729 (bacteria 117727)
    df_1.iloc[row_n,0] = 'add'
    fa_df = pd.merge(fa_data, df_1[['ID', 'biom']], on='ID', how='inner') # 7561*13043
    add_df = pd.DataFrame(fa_df.iloc[row_n, 0:len(fa_df.columns)-1])
    add_df.reset_index(inplace=True)
    add_df.rename(columns=add_df.iloc[0,:], inplace=True)
    add_df.drop([0], inplace=True)
    add_df.reset_index(drop=True, inplace=True) # 13041*2
    add_df['clean_add'] = list(map(lambda x: re.sub('\W+',' ', x), add_df['add']))
    # Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(list(add_df['clean_add']))]
    vec_size = 10
    alpha = 0.025
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    model.build_vocab(tagged_data)
    def emb_(x):
        x1 = x.tags
        x2 = model.docvecs[x1[0]]
        return x2
    add_df['emb'] = list(map(lambda x: emb_(x), tagged_data))

    fa_df_1 = fa_df.drop(index=row_n)
    fa_df_1 = fa_df_1.fillna(0)
    var = len(fa_df.columns)-1

    def multi_(a, b):
        new = []
        idx = []
        for i in range(len(a)):
            try:
                tmp = [j*a[i] for j in b[i]]
                new.append(tmp)
            except:
                idx.append(i)
                new.append(np.array(tmp))
        new = np.array(new)
        return new, idx

    def df_multi(df_a, col_b, var):
        dict = {}
        failed_sample = []
        tmp_df = df_a.iloc[:, 1:var].apply(lambda x: x.astype(float), axis=0)
        for u in range(len(tmp_df)):
            tmp_a = list(tmp_df.iloc[u,:])
            try:
                tmp_final, failed_feature = multi_(tmp_a, col_b)
                dict[df_a['ID'][u]] = tmp_final
            except:
                failed_sample.append(u)
                dict[df_a['ID'][u]] = np.nan
            if u%1000==0:
                print(f'{u} samples done')
        return dict, failed_sample

    my_dict, failed_sample = df_multi(fa_df_1, list(add_df['emb']), var)

    scaler = StandardScaler()
    for k_ in my_dict.keys():
        v_ = my_dict[k_]
        v_1 = scaler.fit_transform(v_)
        my_dict[k_] = v_1
    return my_dict, fa_df_1

def model_input(my_dict, fa_df):
    # prepare data to model
    X = np.array(list(my_dict.values()))
    print(f'shape of embedding dict is {X.shape}')
    X_ = X.reshape(-1,X.shape[1],X.shape[2],1)
    Y = fa_df['biom']
    # convert integers to dummy variables
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    uniqNum_Y = list(set(encoded_Y))
    Y_ = to_categorical(encoded_Y)
    str_Y = encoder.inverse_transform(uniqNum_Y)
    label_dict = dict(zip(uniqNum_Y, str_Y))
    print(f'data for input to model done')
    #split dataset
    x_train, x_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, stratify=Y_, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test, label_dict
