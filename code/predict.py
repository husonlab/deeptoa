import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import pickle
from tensorflow.keras.utils import to_categorical
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import re
import evaluate_metrics
import argparse

def func_mgini(dataFrame):
    df_2 = dataFrame.drop(index=(len(dataFrame)-1))
    '''
    run model on raw dataset
    '''
    # for cluster group dataset
    X = df_2.drop(columns=['ID', 'biom'])
    Y = df_2['biom']

    # encode class value as integers
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    newY = to_categorical(encoded_Y)
    # get encoded mapping list
    encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9])
    label_names = ['Animal_Digestive_system', 'Food_production', 'Freshwater',
                   'Human_Respiratory_system', 'Mammals_Gastrointestinal_tract',
                   'Marine', 'Plants', 'Skin', 'Soil', 'Wastewater']

    # scale data
    scaler_test = RobustScaler()
    x_test = scaler_test.fit_transform(X)

    # reshape dataset
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # assign data format
    x_test = x_test.astype('float64')
    y_test = newY.astype('int')
    return x_test, y_test, label_names


def func_faini(fa_data, mg_data):
    
    fa_data = fa_data.rename(columns={'interpro_accession':'ID'})

    df_mg = mg_data[['ID', 'biom']].drop(index=(len(mg_data)-1))
    print(df_mg)
    fa_df = pd.merge(fa_data, df_mg[['ID', 'biom']], on='ID', how='inner')
    print(f'ini_fa {fa_df}')
    #fa_df = fa_data
    X = fa_df.drop(columns=['ID','biom'])
    X = X.fillna(0)
    Y = fa_df['biom']
    # encode class value as integers
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    # convert integers to dummy variables
    newY = to_categorical(encoded_Y)
    # get encoded mapping list
    encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9])
    label_names = ['Animal_Digestive_system', 'Food_production', 'Freshwater',
                   'Human_Respiratory_system', 'Mammals_Gastrointestinal_tract',
                   'Marine', 'Plants', 'Skin', 'Soil', 'Wastewater']
    scaler_test = RobustScaler()
    x_test = scaler_test.fit_transform(X)
    # reshape dataset
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    # assign data format
    x_test = x_test.astype('float64')
    y_test = newY.astype('int')
    return x_test, y_test, label_names

def func_mgcluster(dataFrame, user, refdata_path, embvec_path, clusterResDict_path):
    def reshape_df(mydf):
        mydf.reset_index(inplace=True)
        mydf.rename(columns=mydf.iloc[0,:], inplace=True)
        mydf.drop([0], inplace=True)
        mydf.reset_index(drop=True, inplace=True)
        return mydf

    def accept_user_data(df_frame, df_user):
        '''
        df_frame = pd.DataFrame(df_frame.iloc[7560,:])
        df_frame = reshape_df(df_frame)
        '''
        df_user.iloc[0:len(df_user)-1, 1:-1] = df_user.iloc[0:len(df_user)-1, 1:-1].astype(float)
        df_user = df_user.T
        df_user = reshape_df(df_user)
        df_tmp = pd.merge(df_frame, df_user, on='ID', how='outer')
        def merge_taxonomy(x, y):
            if x == 'nan':
                x = y
            return x
        df_tmp['taxonomy'] = list(map(lambda x,y: merge_taxonomy(x,y), df_tmp['taxonomy_x'].astype('str'), df_tmp['taxonomy_y'].astype('str')))
        df_tmp.drop(['taxonomy_x', 'taxonomy_y'], axis=1, inplace=True)
        df_tmp = df_tmp.T
        df_tmp = reshape_df(df_tmp)
        biom_  = df_tmp['biom']
        df_tmp.drop('biom', axis=1, inplace=True)
        df_tmp = df_tmp.fillna(0)
        df_tmp['biom'] = biom_
        return df_tmp
    if user == '0':
        df_1 = dataFrame
    elif user == '1':
        '''
        put user's data into our dataframe
        '''
        #refdata_path = '/nfs/wsi/ab/projects/wenhuan/project/stp/DeepToA/test/data/taxfile_framework.csv'
        df_frame = pd.read_csv(refdata_path, low_memory=False)
        df_user = dataFrame
        df_1 = accept_user_data(df_frame, df_user)
    # embedding taxonomy
    var_1 = len(df_1) - 1
    var_2 = len(df_1.columns) - 1
    tax_df = pd.DataFrame(df_1.iloc[var_1, 0:var_2])
    tax_df.reset_index(inplace=True)
    tax_df.rename(columns=tax_df.iloc[0, :], inplace=True)
    tax_df.drop([0], inplace=True)
    tax_df.reset_index(drop=True, inplace=True)

    # features' taxonomy information
    tax_seq = list(map(lambda x: x.split(';'), list(tax_df['taxonomy'])))  # split each record by ;

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
                    '''
                    if value_ in map_dict.keys():
                        new_rec[pos] = map_dict[value_]
                        #print(value_)
                    else:
                        new_rec[pos] = value_
                    '''
                    new_rec[pos] = value_
            except:
                pass
        return new_rec

    tax_seq_0 = list(map(lambda x: fun4rank(x), tax_seq))

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

    tax_df['new_taxonomy'] = tax_seq_1

    # node2vec
    model_n2v = Word2Vec.load(embvec_path)
    print(f'load trained language model done')

    # assigned embedding to bacteria
    # concat method
    failed_idx = []
    res = []
    failed_word=0
    for i in range(len(tax_df)):
        item = tax_df['new_taxonomy'][i]
        lyst = []
        for j in range(len(item)):
            val = str(item[j])
            if val in model_n2v.wv:
                tmp = list(model_n2v.wv[val])
            else:
                print(f'no match {val}')
                tmp = [0] * 10
                failed_word += 1
            '''
            try:
                tmp = list(model_n2v.wv[item[j]])
            except:
                print(f'no match {item[j]}')
                tmp = [0] * 10
            '''
            lyst = lyst + tmp
        if len(lyst) != 80:
            failed_idx.append(i)
        res.append(lyst)
    print(f'failed assigned vector word number {failed_word}')

    print(f'taxonomy embedding done')
    # different cluster method
    matrix_res = np.matrix(res)
    print (f'start building clustering model')

    if user == '0':
        # AGNES
        agnes_clf = AgglomerativeClustering(n_clusters=10000)
        agnes_clf.fit(matrix_res)
        agnes_hat = agnes_clf.labels_
        tax_df['AGNES'] = agnes_hat

        var = 'AGNES'
        dict_ = {}
        for i in range(len(tax_df)):
            if tax_df[var][i] not in dict_.keys():
                dict_[tax_df[var][i]] = [tax_df['ID'][i]]
            else:
                dict_[tax_df[var][i]].append(tax_df['ID'][i])

        print(len(dict_.keys()))
        print(f'assigned each feature to calculated group according to {var} method done')

    elif user == '1':
        print(tax_df.columns)
        emb_list = res
        tax_df['emb'] = emb_list
        ini_rec = np.array(emb_list[0:117727])
        new_rec = np.array(emb_list[117727:])

        dis = cdist(new_rec, ini_rec, metric='cosine')
        simi = np.argmax(dis, axis=1) # for each new record, the index of otu id in initial dataset that has highest similarity
        # map dict for initial feature and the new feature, which are most similar with each other
        new_rec_df  = tax_df.iloc[117727:,:]
        new_rec_df.reset_index(drop=True, inplace=True)
        ini_id = list(map(lambda x: tax_df['ID'][x], list(simi)))
        new_rec_df['ini_id'] = ini_id
        new_ini_dict = dict(zip(new_rec_df['ID'], new_rec_df['ini_id']))

        #with open('/nfs/wsi/ab/projects/wenhuan/project/stp/mg_data/v2_ncbi_gtdb_group_dict.pkl', 'rb') as tf:
        with open(clusterResDict_path, 'rb') as tf:
            dict_ = pickle.load(tf)

        inverse_dict = {} # key is ini otu id, value is group
        for key, val in dict_.items():
            for sub_val in val:
                inverse_dict[sub_val] = key

        for i in range(len(new_ini_dict)):
            dict_[inverse_dict[list(new_ini_dict.values())[i]]].append(list(new_ini_dict.keys())[i])

    # merge dataframe by group
    sub_df = df_1[['ID', 'biom']]
    sub_df.drop(index=var_1, inplace=True)
    cor_df = df_1.iloc[0:var_1, 1:var_2-1]

    failed_rec = []
    failed_group = []
    for k in dict_.keys():
        feature_set = dict_[k]
        tmp_col = np.zeros(var_1, )
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

    # for cluster group dataset
    X = sub_df.drop(columns=['ID', 'biom'])
    Y = sub_df['biom']
    # encode class value as integers
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    # convert integers to dummy variables
    newY = to_categorical(encoded_Y)
    # get encoded mapping list
    encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9])
    label_names = ['Animal_Digestive_system', 'Food_production', 'Freshwater',
                   'Human_Respiratory_system', 'Mammals_Gastrointestinal_tract',
                   'Marine', 'Plants', 'Skin', 'Soil', 'Wastewater']
    x_test = X
    y_test = newY
    scaler_test = RobustScaler()
    x_test = scaler_test.fit_transform(x_test)

    # reshape dataset
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    print(f'the shape of training dataset {x_test.shape}')
    x_test = x_test.astype('float64')
    y_test = y_test.astype('int')
    return x_test, y_test, label_names

def func_faEmb(fa_data, mg_data):
    fa_data = fa_data.rename(columns={'interpro_accession':'ID'})
    fa_data.iloc[fa_data.shape[0]-1,0] = 'add' 
    df_1 = mg_data
    df_1.loc[df_1.shape[0]-1] = 'add'
    fa_df = pd.merge(fa_data, df_1[['ID','biom']], on='ID', how='inner') # 7561*13043, 1513*13043
    print (fa_df)

    add_df = pd.DataFrame(fa_df.iloc[fa_df.shape[0]-1, 0:len(fa_df.columns)-1])
    print(add_df)
    add_df.reset_index(inplace=True)
    add_df.rename(columns=add_df.iloc[0,:], inplace=True)
    add_df.drop([0], inplace=True)
    add_df.reset_index(drop=True, inplace=True) # 13041*2
    add_df['clean_add'] = list(map(lambda x: re.sub('\W+',' ', x), add_df['add']))
    # Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(list(add_df['clean_add']))]

    max_epochs = 100
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

    fa_df_1 = fa_df.drop(index=fa_df.shape[0]-1)
    fa_df_1 = fa_df_1.fillna(0)
    var = len(fa_df.columns)-1

    def multi_(a, b):
        new = []
        idx = []
        for i in range(len(a)):
            try:
                tmp = [j*a[i] for j in b[i]]
                #print(tmp)
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
            if u%500==0:
                print(f'{u} samples done')
        return dict, failed_sample

    my_dict, failed_sample = df_multi(fa_df_1, list(add_df['emb']), var)

    scaler = StandardScaler()
    for k_ in my_dict.keys():
        v_ = my_dict[k_]
        v_1 = scaler.fit_transform(v_)
        my_dict[k_] = v_1

    print (f'embedding dict done')

    # prepare data to model
    x_test = np.array(list(my_dict.values()))
    print(f'length of embedding dict is {len(x_test)}')
    x_test = x_test.reshape(-1,13041,10,1)
    y_test = fa_df_1['biom']

    # encode class value as integers
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y_test)
    y_test = to_categorical(encoded_Y)
    # get encoded mapping list
    encoder.inverse_transform([0,1,2,3,4,5,6,7,8,9])
    label_names = ['Animal_Digestive_system', 'Food_production', 'Freshwater',
                   'Human_Respiratory_system', 'Mammals_Gastrointestinal_tract',
                   'Marine', 'Plants', 'Skin', 'Soil', 'Wastewater']
    print(f'data for input to model done')

    return x_test, y_test, label_names

def func_ensemble(fa_data, mg_data):
    mg_x_test, mg_y_test, label_names = func_mgcluster(mg_data)
    fa_x_test, fa_y_test, label_names_1 = func_faini(fa_data, mg_data)
    x_test = [mg_x_test, fa_x_test]
    y_test = mg_y_test
    return x_test, y_test, label_names

def func_deeptoa(fa_data, mg_data, user):
    mg_x_test, mg_y_test, label_names = func_mgcluster(mg_data, user)
    fa_x_test, fa_y_test, label_names_1 = func_faEmb(fa_data, mg_data)
    x_test = [mg_x_test, fa_x_test]
    y_test = mg_y_test
    return x_test, y_test, label_names


def dataGenerator(dataPath, testid_path, modelmode, pretrainedModelPath, fadataPath, rocPath, cmPath, refdata_path=None, embvec_path=None, clusterResDict_path=None, user=None):
    mgdf = pd.read_csv(dataPath, low_memory=False)  # shape 7561*120022 (bacteria 120022)
    test_idx = pd.read_csv(testid_path)
    testdf = pd.merge(mgdf.iloc[0:(mgdf.shape[0])-1, :], test_idx, on='ID', how='right')
    testdf.loc[testdf.shape[0]] = mgdf.loc[mgdf.shape[0]-1]
    if modelmode == 'mg_ini':
        x_test, y_test, label_names = func_mgini(testdf)
        evaluate_metrics.ensemble_metrics_caculation(pretrainedModelPath, x_test, y_test, label_names, rocPath, cmPath)
    elif modelmode == 'fa_ini':
        fa_data = pd.read_csv(fadataPath, low_memory=False)
        x_test, y_test, label_names = func_faini(fa_data, testdf)
        evaluate_metrics.ensemble_metrics_caculation(pretrainedModelPath, x_test, y_test, label_names, rocPath, cmPath)
    elif modelmode == 'mg_cluster':
        x_test, y_test, label_names = func_mgcluster(testdf, user, refdata_path, embvec_path, clusterResDict_path)
        evaluate_metrics.ensemble_metrics_caculation(pretrainedModelPath, x_test, y_test, label_names, rocPath, cmPath)
    elif modelmode == 'fa_emb':
        fa_data = pd.read_csv(fadataPath, low_memory=False)
        x_test, y_test, label_names = func_faEmb(fa_data, testdf)
        evaluate_metrics.ensemble_metrics_caculation(pretrainedModelPath, x_test, y_test, label_names, rocPath, cmPath)
    elif modelmode == 'ensemble':
        fa_data = pd.read_csv(fadataPath, low_memory=False)
        x_test, y_test, label_names = func_ensemble(fa_data, testdf)
        evaluate_metrics.ensemble_metrics_caculation(pretrainedModelPath, x_test, y_test, label_names, rocPath, cmPath)
    elif modelmode == 'deeptoa':
        fa_data = pd.read_csv(fadataPath, low_memory=False)
        x_test, y_test, label_names = func_deeptoa(fa_data, testdf, user)
        evaluate_metrics.ensemble_metrics_caculation(pretrainedModelPath, x_test, y_test, label_names, rocPath, cmPath)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mm','--modelMode',
                        dest = 'model_mode',
                        required = True,
                        help = 'assign prediction model type ')
    parser.add_argument('-mdf','--mainDF',
                        dest = 'maindata_path',
                        required = True,
                        help = 'load main data')
    parser.add_argument('-idx','--idxFile',
                        dest = 'testid_path',
                        required = True,
                        help = 'load test set ID index')
    parser.add_argument('-inifa','--inifaDF',
                        dest = 'inifa_path',
                        required = False,
                        help = 'functional profile without biom information')
    parser.add_argument('-profa','--processedfaDF',
                        dest = 'profa_path',
                        required = False,
                        help = 'functional profile with description, without biom information')
    parser.add_argument('-u','--user',
                        dest = 'user',
                        required = True,
                        default='0',
                        help = 'user upload own data')
    parser.add_argument('-mp','--modelPath',
                        dest = 'model_path',
                        required = True,
                        default='0',
                        help = 'pretrained model path')
    parser.add_argument('-rp','--rocPath',
                        dest = 'roc_path',
                        required = True,
                        default='0',
                        help = 'roc saving path')
    parser.add_argument('-cp','--cmPath',
                        dest = 'cm_path',
                        required = True,
                        default='0',
                        help = 'confusion matrix saving path')
    parser.add_argument('-refdf','--refdataPath',
                        dest = 'refdata_path',
                        required = False,
                        default = None,
                        help = 'reference dataframe available while user=1')
    parser.add_argument('-ev','--embvecPath',
                        dest = 'emb_path',
                        required = False,
                        default = None,
                        help = 'path for taxonomy embedding vector')
    parser.add_argument('-cresd','--clusterResPath',
                        dest = 'args.clusterResDict_path',
                        required = False,
                        default = None,
                        help = 'cluster assignment available while user=1')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_path = args.maindata_path
    testid_path = args.testid_path
    model_path = args.model_path
    if args.model_mode == 'mg_ini':
        fadata_path = None
    elif args.model_mode == 'fa_ini':
        fadata_path = args.inifa_path
    elif args.model_mode == 'mg_cluster':
        user = args.user
        fadata_path = None
        refdata_path = args.refdata_path
        embvec_path = args.emb_path
        clusterResDict_path = args.clusterResDict_path
    elif args.model_mode == 'fa_emb':
        fadataPath = args.profa_path
    elif args.model_mode == 'ensemble':
        data_path = args.maindata_path
        fadata_path = args.inifa_path
    elif args.model_mode == 'deeptoa':
        fadata_path = args.profa_path
    dataGenerator(data_path, testid_path, args.model_mode, model_path, fadata_path, args.roc_path, args.cm_path, refdata_path, embvec_path, clusterResDict_path, user)




