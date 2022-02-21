'''
Predict sample source by processed taxonomic profile, processed functional profile or ensemble model.
'''
import argparse
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import taxonomic_profile
import functional_profile

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     """ Welcome to DeepToA.
                                     An ensemble deep learning framework for predicting the Theater of Activity ToA or source (for microbial community)
                                     Specific instructions can be found in README.md file.
                                     >> example commandline （User's data, ensemble model）
                                     >> ./DeepToA.py python code/main.py
                                        -ot ./data/test4user_withid.csv -dt ./data/data_framework.csv -mf ./data/ncbi_map_gtdb_long.csv
                                        -e model/mapped_mg_gtdb_taxonomy_node2vec_model.vec  -f ./data/T_interpro_additional.csv
                                        -df ./data/interpro/interpro_frame.csv -em ./model/DeepToA.h5
                                        -m e -l ./log/log.txt -c 10000 -u '1' -sm '0'
                                        """,
                                     formatter_class = argparse.RawTextHelpFormatter, add_help=False)

    parser.add_argument("-ot", "--own taxonomy",
                        dest="taxdata_path",
                        required=False,
                        type=str,
                        help="user's initial taxonomic profile path")
    parser.add_argument("-dt", "--developer taxonomy",
                        dest="refdata_path",
                        required=False,
                        type=str,
                        help="developer's initial taxonomic profile path")
    parser.add_argument("-mf", "--mapping",
                        dest="mapping_file",
                        required=False,
                        metavar="FILE",
                        help="mapping file path, map ncbi name to gtdb standard")
    parser.add_argument("-u", "--use the dataset offer by us",
                        dest="user_define",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="if user_define=0, use taxonomic profile offered by us. Or use your own file")
    parser.add_argument("-e", "--embeddingVector",
                        dest="emb_path",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="file path of pre-trained embedding vector")
    parser.add_argument("-c", "--clusterGroup",
                        dest="cluster_num",
                        action="store",
                        required=False,
                        type=int,
                        metavar='\b',
                        help="group number for cluster method, default is 10,000")
    parser.add_argument("-tm", "--tp_profile",
                        dest="tax_modelpath",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="model path of trained taxonomic classification model")
    parser.add_argument("-f", "--functional_profile",
                        dest="fadata_path",
                        required=False,
                        type=str,
                        help="initial function profile path")
    parser.add_argument("-df", "--developer_functional_profile_framework",
                        dest="interpro_frame_path",
                        required=False,
                        type=str,
                        help="path of developer's functional profile framework")
    parser.add_argument("-fm", "--fp_model",
                        dest="fun_modelpath",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="model path of classification model trained on processed functional profile")
    parser.add_argument("-em", "--ensemble_model",
                        dest="en_modelpath",
                        action="store",
                        required=False,
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
    parser.add_argument("-sm", "--process taxonomic data",
                        dest="self_map",
                        action="store",
                        required=False,
                        metavar='\b',
                        help="if self_map=1, user have processed their taxonomic data as our format")



    args = parser.parse_args()
    if args.model == 't':
        if args.user_define == '0':
            main_df, tax_df, row_num, col_num = taxonomic_profile.process_ds(args.taxdata_path, args.mapping_file, args.user_define, args.self_map)
            emb_matrix = taxonomic_profile.text2num(tax_df, args.emb_path)
            dr_df = taxonomic_profile.generate_ds(emb_matrix, tax_df, main_df, row_num, col_num, args.cluster_num)
            x_train, y_train, x_val, y_val, x_test, y_test, label_dict_ = taxonomic_profile.model_input(dr_df)
            tf.compat.v1.disable_v2_behavior()
            tax_model = load_model(args.tax_modelpath)
            prediction = tax_model.predict(x_test)
            real_label = y_test
        elif args.user_define == '1':
            main_df, tax_df, row_num, col_num = taxonomic_profile.process_ds(args.taxdata_path, args.mapping_file, args.user_define, args.self_map, args.refdata_path)
            emb_matrix = taxonomic_profile.text2num(tax_df, args.emb_path)
            dr_df = taxonomic_profile.generate_ds(emb_matrix, tax_df, main_df, row_num, col_num, args.cluster_num)
            x_, y_, label_dict_ = taxonomic_profile.model_input4user(dr_df)
            tf.compat.v1.disable_v2_behavior()
            tax_model = load_model(args.tax_modelpath)
            prediction = tax_model.predict(x_)
            real_label = y_

    elif args.model == 'f':
        if args.user_define == '0':
            fa_dict, ini_fa_df = functional_profile.process_ds(args.fadata_path, args.taxdata_path, args.user_define)
            fa_x_train, fa_y_train, fa_x_val, fa_y_val, fa_x_test, fa_y_test, label_dict_ = functional_profile.model_input(fa_dict, ini_fa_df)
            tf.compat.v1.disable_v2_behavior()
            fa_model = load_model(args.fun_modelpath)
            prediction = fa_model.predict(fa_x_test)
            real_label = fa_y_test
        if args.user_define == '1':
            fa_dict, ini_fa_df = functional_profile.process_ds(args.fadata_path, args.taxdata_path, args.user_define, args.interpro_frame_path)
            x_, y_, label_dict_ = functional_profile.model_input4user(fa_dict, ini_fa_df)
            tf.compat.v1.disable_v2_behavior()
            fa_model = load_model(args.fun_modelpath)
            prediction = fa_model.predict(x_)
            real_label = y_



    elif args.model == 'e':
        if args.user_define == '0':
            main_df, tax_df, row_num, col_num = taxonomic_profile.process_ds(args.taxdata_path, args.mapping_file, args.user_define)
            emb_matrix = taxonomic_profile.text2num(tax_df, args.emb_path)
            dr_df_0 = taxonomic_profile.generate_ds(emb_matrix, tax_df, main_df, row_num, col_num, args.cluster_num)
            x_train, y_train, x_val, y_val, x_test, y_test, label_dict_ = taxonomic_profile.model_input(dr_df_0)
            fa_dict, ini_fa_df = functional_profile.process_ds(args.fadata_path, args.taxdata_path, args.user_define)
            fa_x_train, fa_y_train, fa_x_val, fa_y_val, fa_x_test, fa_y_test, fa_label_dict_ = functional_profile.model_input(fa_dict, ini_fa_df)
            tf.compat.v1.disable_v2_behavior()
            ensemble_model = load_model(args.en_modelpath)
            prediction = ensemble_model.predict([x_test, fa_x_test])
            loss, acc = ensemble_model.evaluate([x_test, fa_x_test], y_test)
            print(acc)
            real_label = y_test
        elif args.user_define == '1':
            main_df, tax_df, row_num, col_num = taxonomic_profile.process_ds(args.taxdata_path, args.mapping_file, args.user_define, args.self_map, args.refdata_path)
            emb_matrix = taxonomic_profile.text2num(tax_df, args.emb_path)
            dr_df = taxonomic_profile.generate_ds(emb_matrix, tax_df, main_df, row_num, col_num, args.cluster_num)
            x_, y_, label_dict_ = taxonomic_profile.model_input4user(dr_df)
            fa_dict, ini_fa_df = functional_profile.process_ds(args.fadata_path, args.taxdata_path, args.user_define, args.interpro_frame_path)
            fa_x_, fa_y_, fa_label_dict_ = functional_profile.model_input4user(fa_dict, ini_fa_df)
            tf.compat.v1.disable_v2_behavior()
            ensemble_model = load_model(args.en_modelpath)
            prediction = ensemble_model.predict([x_, fa_x_])
            print((y_==fa_y_).all())
            real_label = y_
        else:
            print('error')
    else:
        print('error')
        

    # export prediction result
    logger = open(args.result, 'w')
    logger.write('probability' + '\t' + 'pre_class' + '\t' + 'real_class'+ '\t' + 'pre_class_str' + '\t' + 'real_class_str' + '\n')
    print(real_label[0])
    ini_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ini_str = ['Animal_Digestive_system', 'Food_production', 'Freshwater','Human_Respiratory_system',
            'Mammals_Gastrointestinal_tract', 'Marine','Plants', 'Skin', 'Soil', 'Wastewater']
    ini_label_dict = dict(zip(ini_num, ini_str))
    print(ini_label_dict)
    for i in range(prediction.shape[0]):
        probability_ = prediction[i]
        class_ = np.argmax(probability_)
        label_ = np.argmax(real_label[i])
        if args.user_define == '0':
            class_str = label_dict_[class_]
        elif args.user_define == '1':
            class_str = ini_label_dict[class_]
        label_str = label_dict_[label_]
        logger.write(str(probability_) + '\t' + str(class_) + '\t' + str(label_) + '\t' + str(class_str) + '\t' + str(label_str)+ '\n')
    logger.close()








