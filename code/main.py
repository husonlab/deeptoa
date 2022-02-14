'''
build model on processed taxonomic profile
'''
import os, argparse
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Lambda, Conv2D, Activation, MaxPooling2D, Convolution2D,LSTM, Dropout, Input, add, Bidirectional
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import datetime
import taxonomic_profile
import functional_profile

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """ Welcome to DeepToA.
                                     An ensemble deep learning framework for predicting the source of ToA(microbial community)
                                     Specific instructions can be found in README.md file.
                                     >> example commandline
                                     >> ./DeepToA.py -m examples/T2D/T2D_datasets.txt -y examples/T2D/T2D_ylab.txt -t examples/T2D -e1 30 -e2 10 -l examples/results/T2D_summarise """,
                                     formatter_class = argparse.RawTextHelpFormatter, add_help=False)

    parser.add_argument("-t", "--taxonomy",
                        dest="taxdata_path",
                        required=True,
                        type=str,
                        help="initial taxonomic profile path")
    parser.add_argument("-mf", "--mapping",
                        dest="mapping_file",
                        required=True,
                        metavar="FILE",
                        help="mapping file path, map ncbi name to gtdb standard")
    parser.add_argument("-u", "--use the dataset offer by us",
                        dest="map_stat",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="if map=1, use taxonomic profile offered by us. Or use your own file")
    parser.add_argument("-e", "--embeddingVector",
                        dest="emb_path",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="file path of pre-trained embedding vector")
    '''
    parser.add_argument("-c", "--clusterGroup",
                        dest="cluster_num",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="group number for cluster method, default is 10,000")
   '''
    parser.add_argument("-tm", "--tp_profile",
                        dest="tax_modelpath",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="model path of trained taxonomic classification model")
    parser.add_argument("-f", "--functional_profile",
                        dest="fadata_path",
                        required=True,
                        type=str,
                        help="initial function profile path")
    parser.add_argument("-fm", "--fp_model",
                        dest="fun_modelpath",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="model path of classification model trained on processed functional profile")
    parser.add_argument("-em", "--ensemble_model",
                        dest="en_modelpath",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="model path of classification model trained on processed functional profile")
    parser.add_argument("-m", "--modelChoose",
                        dest="model",
                        action="store",
                        required=True,
                        metavar='\b',
                        help="t build model for taxonomic profile. f build model for functional profile. e ensemble model")
    parser.add_argument("-l", "--log_path",
                        dest="result",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="result file")


    args = parser.parse_args()
    if args.model == 't':
        main_df, tax_df, row_num, col_num = taxonomic_profile.process_ds(args.taxdata_path, args.mapping_file)
        emb_matrix = taxonomic_profile.text2num(tax_df, args.emb_path)
        dr_df = taxonomic_profile.generate_ds(emb_matrix, tax_df, main_df, row_num, col_num)
        x_train, y_train, x_val, y_val, x_test, y_test, label_dict_ = taxonomic_profile.model_input(dr_df)
        tf.compat.v1.disable_v2_behavior()
        tax_model = load_model(args.tax_modelpath)
        prediction = tax_model.predict(x_test)
        real_label = y_test

    elif args.model == 'f':
        fa_dict, ini_fa_df = functional_profile.process_ds(args.fadata_path, args.taxdata_path)
        fa_x_train, fa_y_train, fa_x_val, fa_y_val, fa_x_test, fa_y_test, label_dict_ = functional_profile.model_input(fa_dict, ini_fa_df)
        tf.compat.v1.disable_v2_behavior()
        fa_model = load_model(args.fun_modelpath)
        prediction = fa_model.predict(fa_x_test)
        real_label = fa_y_test

    elif args.model == 'e':
        main_df, tax_df, row_num, col_num = taxonomic_profile.process_ds(args.taxdata_path, args.mapping_file)
        emb_matrix = taxonomic_profile.text2num(tax_df, args.emb_path)
        dr_df = taxonomic_profile.generate_ds(emb_matrix, tax_df, main_df, row_num, col_num)
        x_train, y_train, x_val, y_val, x_test, y_test, label_dict_ = taxonomic_profile.model_input(dr_df)
        fa_dict, ini_fa_df = functional_profile.process_ds(args.fadata_path, args.taxdata_path)
        fa_x_train, fa_y_train, fa_x_val, fa_y_val, fa_x_test, fa_y_test = functional_profile.model_input(fa_dict, ini_fa_df)
        tf.compat.v1.disable_v2_behavior()
        ensemble_model = load_model(args.en_modelpath)
        prediction = ensemble_model.predict([x_test, fa_x_test])
        real_label = y_test

        

    # export prediction result
    logger = open(args.result, 'w')
    logger.write('probability' + '\t' + 'pre_class' + '\t' + 'real_class'+ '\t' + 'pre_class_str' + '\t' + 'real_class_str' + '\n')
    print(real_label[0])
    for i in range(prediction.shape[0]):
        probability_ = prediction[i]
        class_ = np.argmax(probability_)
        label_ = np.argmax(real_label[i])
        class_str = label_dict_[class_]
        label_str = label_dict_[label_]
        if i == 0:
            print(f'probability is {probability_}, class_ is {class_}, label_ is {label_}, class is {class_str}, label is {label_str}')
        logger.write(str(probability_) + '\t' + str(class_) + '\t' + str(label_) + '\t' + str(class_str) + '\t' + str(label_str)+ '\n')
    logger.close()






