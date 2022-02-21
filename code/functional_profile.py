'''
prepare functional profile for building classification model
using interpro abundance table to build the classification model
build model on the dataset with description
'''
import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

'''
Function for accept user's dataset
'''
def reshape_df(mydf):
    mydf.reset_index(inplace=True)
    mydf.rename(columns=mydf.iloc[0,:], inplace=True)
    mydf.drop([0], inplace=True)
    mydf.reset_index(drop=True, inplace=True)
    return mydf

def accept_user_data(df_frame, df_user):
    df_user = df_user.T
    df_user = reshape_df(df_user)
    # except ID, biom, the other columns need be 'float' type
    df_user.iloc[0:df_user.shape[0]-1, 1:df_user.shape[1]-1] = df_user.iloc[0:df_user.shape[0]-1, 1:df_user.shape[1]-1].astype('float')
    #df_user = df_user.rename(columns={'add':'description'})
    df_tmp = pd.merge(df_frame, df_user, on='ID', how='outer')
    def merge_cols(x, y):
        if x == 'nan':
            x = y
        return x
    df_tmp['description'] = list(map(lambda x,y: merge_cols(x,y), df_tmp['description_x'].astype('str'), df_tmp['description_y'].astype('str')))
    df_tmp.drop(['description_x', 'description_y'], axis=1, inplace=True)
    df_tmp = df_tmp.T
    df_tmp = reshape_df(df_tmp)
    biom_  = df_tmp['biom']
    df_tmp.drop('biom', axis=1, inplace=True)
    df_tmp = df_tmp.fillna(0)
    df_tmp['biom'] = biom_
    # control col number to 13041
    col_current = df_tmp.shape[1]-1
    df_row = df_tmp.shape[0]-1
    frame_cnt = df_frame.shape[0]-1
    df_sub = df_tmp.iloc[0:df_row,1:col_current]
    col_del_num = col_current - 1 - frame_cnt
    col_sum = pd.DataFrame(df_sub.sum())
    col_sum.reset_index(inplace=True)
    col_sum.rename(columns={'index':'feature', 0: 'cnt'}, inplace=True)
    zero_col = col_sum.drop(col_sum[col_sum['cnt'] != 0].index)
    list_tmp = list(zero_col['feature'])
    idx = random.sample(list_tmp, col_del_num)
    df_tmp = df_tmp.drop(idx, axis=1)
    return df_tmp

'''
Assigned samples' label to functional profile by incooperating with taxonomic profile.
Embedding description data of each InterPro ID.
Expanded each sample's size with numeric description data.
'''
def process_ds(fa_path, tax_path, user_define='1', interpro_frame_path=None):
    fa_data = pd.read_csv(fa_path,low_memory=False)
    fa_data = fa_data.rename(columns={'interpro_accession':'ID'})
    row_n = len(fa_data)-1
    fa_data.iloc[row_n,0] = 'description'
    df_1 = pd.read_csv(tax_path, low_memory=False)  # shape 7561*117729 (bacteria 117727)
    df_1.iloc[row_n,0] = 'description'
    fa_df = pd.merge(fa_data, df_1[['ID', 'biom']], on='ID', how='inner') # 7561*13043
    print(user_define)
    if user_define == '1':
        df_frame = pd.read_csv(interpro_frame_path,low_memory=False)
        fa_df = accept_user_data(df_frame, fa_df)
        print(fa_df)
    df_row = fa_df.shape[0]-1
    df_col = fa_df.shape[1]-1
    add_df = pd.DataFrame(fa_df.iloc[df_row, 0:df_col])
    add_df = reshape_df(add_df)
    add_df['clean_add'] = list(map(lambda x: re.sub('\W+',' ', x), add_df['description']))
    print('start embedding description')
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
    if user_define == '1':
        fa_df_1 = fa_df_1.sort_values(by='ID', ascending=True)
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
        dict_ = {}
        failed_sample = []
        tmp_df = df_a.iloc[:, 1:var].apply(lambda x: x.astype(float), axis=0)
        for u in range(len(tmp_df)):
            tmp_a = list(tmp_df.iloc[u,:])
            try:
                tmp_final, failed_feature = multi_(tmp_a, col_b)
                dict_[df_a['ID'][u]] = tmp_final
            except:
                failed_sample.append(u)
                dict_[df_a['ID'][u]] = np.nan
            if u%1000==0:
                print(f'{u} samples done')
        return dict_, failed_sample

    my_dict, failed_sample = df_multi(fa_df_1, list(add_df['emb']), var)

    scaler = StandardScaler()
    for k_ in my_dict.keys():
        v_ = my_dict[k_]
        v_1 = scaler.fit_transform(v_)
        my_dict[k_] = v_1
    return my_dict, fa_df_1

'''
Process dataframe to the format which can be accept by model.
For testing the data offered by us.
'''
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

'''
Process dataframe to the format which can be accept by model.
For taking user-generated data as input
'''
def model_input4user(my_dict, fa_df):
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
    print(f'prepare data offered by user for input to model done')
    return X_, Y_, label_dict
