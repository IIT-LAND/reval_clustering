# EXAMPLE 4: UCI dataset classifier/clustering combinations
import csv
import re
import numpy as np
import os
from collections import namedtuple


def build_ucidatasets():
    """
    Function that loads 18 different benchmark dataset for clustering/classification
    from the UCI Machine Learning Repository.

    :return: named tuple with fields name the loaded datasets
    :rtype: namedtuple
    """
    UCI_datasets = namedtuple('UCI_datasets', ['biodeg',
                                               'breastwi',
                                               'climate',
                                               'banknote',
                                               'ecoli',
                                               'glass',
                                               'ionosphere',
                                               'iris',
                                               'liver',
                                               'movement',
                                               'parkinsons',
                                               'seeds',
                                               'transfusion',
                                               'wholesale',
                                               'yeast',
                                               'forest',
                                               'urban',
                                               'leaf'])

    # Datasets
    file_names = ['biodeg.csv',
                  'breast-cancer-wisconsin.data',
                  'climate.dat',
                  'data_banknote_authentication.txt',
                  'ecoli.data',
                  'glass.data',
                  'ionosphere.data',
                  'iris.data',
                  'liver.data',
                  'movement_libras.data',
                  'parkinsons.data',
                  'seeds_dataset.txt',
                  'transfusion.data',
                  'Wholesale_customers_data.csv',
                  'yeast.data',
                  'forest_tr.csv', 'forest_ts.csv',
                  'urban_tr.csv', 'urban_ts.csv',
                  'leaf.csv']

    # Read datasets
    data_dict = {}
    for fn in file_names:
        name_dataset = fn.split('.')[0]
        if name_dataset == 'biodeg':
            with open(os.path.join(os.getcwd(), 'working_examples/datasets', fn)) as f:
                rd = csv.reader(f, delimiter=';')
                data = []
                for r in rd:
                    data.append(r)
            data_dict[name_dataset] = data
        elif re.search('seed', name_dataset):
            with open(os.path.join(os.getcwd(), 'working_examples/datasets', fn)) as f:
                rd = csv.reader(f, delimiter='\t')
                data = []
                for r in rd:
                    data.append(r)
            data_dict[name_dataset] = data
        else:
            with open(os.path.join(os.getcwd(), 'working_examples/datasets', fn)) as f:
                rd = csv.reader(f)
                data = []
                for r in rd:
                    data.append(r)
            data_dict[name_dataset] = data

    # Biodeg
    classlab_biodeg = {'RB': 1,
                       'NRB': 0}  # RB: ready biodegradable, NRB: not ready biodegradable
    biodeg = {'data': np.array([vect[:-1] for vect in data_dict['biodeg']]).astype(float),
              'target': np.array([classlab_biodeg[vect[-1]] for vect in data_dict['biodeg']]).astype(int),
              'description': "Data set containing values for 41 attributes (molecular descriptors) used to "
                             "classify 1055 chemicals into 2 classes, i.e., ready (1) and not ready (0) biodegradable."}

    # Breast cancer WI
    classlab_bc = {'2': 0,
                   '4': 1}  # 2: benign, 4: malignant
    data_bc = np.array([vect[1:-1] for vect in data_dict['breast-cancer-wisconsin']])
    target_bc = np.array([classlab_bc[vect[-1]] for vect in data_dict['breast-cancer-wisconsin']]).astype(int)
    drop_idx = []
    for idx, val in enumerate(data_bc):
        try:
            chk = list(val).index('?')
            drop_idx.append(idx)
        except ValueError:
            pass
    breastwi = {'data': np.delete(data_bc, drop_idx, 0).astype(float),
                'target': np.delete(target_bc, drop_idx),
                'description': "Original Wisconsin Breast Cancer Database. 0 benign, 1 malignant"}

    # Climate
    data_cl, target_cl = [], []
    for vect in data_dict['climate']:
        if vect[-1] == '':
            data_cl.append(vect[2:len(vect) - 2])
            target_cl.append(vect[len(vect) - 2])
        else:
            data_cl.append(vect[2:-1])
            target_cl.append(vect[-1])
    climate = {'data': np.array(data_cl).astype(float),
               'target': np.array(target_cl).astype(int),
               'description': "Given Latin hypercube samples of 18 climate model input parameter values, "
                              "predict climate model simulation crashes and determine the parameter "
                              "value combinations that cause the failures. "
                              "Simulation outcome (0 = failure, 1 = success)."}

    # Banknote
    banknote = {'data': np.array([vect[:-1] for vect in data_dict['data_banknote_authentication']]).astype(float),
                'target': np.array([vect[-1] for vect in data_dict['data_banknote_authentication']]).astype(int),
                'description': "Data were extracted from images that were taken from genuine and forged "
                               "banknote-like specimens. For digitization, an industrial camera usually "
                               "used for print inspection was used. The final images have 400x 400 pixels. "
                               "Due to the object lens and distance to the investigated object "
                               "gray-scale pictures with a resolution of about 660 dpi were gained. "
                               "Wavelet Transform tool were used to extract features from images. Two classes 0/1."}

    # Ecoli
    classlab_ecoli = {'cp': 0,
                      'im': 1,
                      'imL': 2,
                      'imS': 3,
                      'imU': 4,
                      'om': 5,
                      'omL': 6,
                      'pp': 7}
    ecoli = {'data': np.array([vect[1:-1] for vect in data_dict['ecoli']]).astype(float),
             'target': np.array([classlab_ecoli[vect[-1]] for vect in data_dict['ecoli']]).astype(int),
             'description': "This data contains protein localization sites. 8 classes 0-7."}

    # Glass
    classlab_glass = {0: 0,
                      1: 1,
                      2: 2,
                      4: 3,
                      5: 4,
                      6: 5}
    glass = {'data': np.array([vect[1:-1] for vect in data_dict['glass']]).astype(float),
             'target': np.array([classlab_glass[(int(vect[-1]) - 1)] for vect in data_dict['glass']]).astype(int),
             'description': "From USA Forensic Science Service; 6 types of glass; defined in terms of their "
                            "oxide content (i.e. Na, Fe, K, etc). "
                            "Type of glass: 0 building_windows_float_processed; "
                            "1 building_windows_non_float_processed; "
                            "2 vehicle_windows_float_processed; "
                            "3 containers; 4 tableware; 5 headlamps"}

    # Ionosphere
    classlab_iono = {'g': 0,
                     'b': 1}  # g: good, b: bad
    ionosphere = {'data': np.array([vect[:-1] for vect in data_dict['ionosphere']]).astype(float),
                  'target': np.array([classlab_iono[vect[-1]] for vect in data_dict['ionosphere']]).astype(int),
                  'description': "Classification of radar returns from the ionosphere. Classes good (0), bad (1)."}

    # Iris
    classlab_iris = {'Iris-setosa': 0,
                     'Iris-versicolor': 1,
                     'Iris-virginica': 2}
    iris = {'data': np.array([vect[:-1] for vect in data_dict['iris'][:-1]]).astype(float),
            'target': np.array([classlab_iris[vect[-1]] for vect in data_dict['iris'][:-1]]).astype(int),
            'description': "Fisher's Iris dataset. 0 Setosa, 1 Versicolor, 2 Virginica."}

    # Liver
    classlab_liver = {'0.0': 0,
                      '0.5': 0,
                      '1.0': 0,
                      '10.0': 1,
                      '12.0': 1,
                      '15.0': 1,
                      '16.0': 1,
                      '2.0': 0,
                      '20.0': 1,
                      '3.0': 1,
                      '4.0': 1,
                      '5.0': 1,
                      '6.0': 1,
                      '7.0': 1,
                      '8.0': 1,
                      '9.0': 1}
    liver = {'data': np.array([vect[0:5] for vect in data_dict['liver']]).astype(float),
             'target': np.array([classlab_liver[vect[5]] for vect in data_dict['liver']]).astype(int),
             'description': "Liver disorders dataset. Blood test values in features. "
                            "Each line in the dataset constitutes the record of a single male individual. "
                            "Dichotomous classes based on drinks number < 3.0 (0), >=3.0 (1), for reference see "
                            "(Turney, 1995, Cost-sensitive classification: Empirical evaluation of a hybrid "
                            "genetic decision tree induction algorithm). "
                            "Drinks number was originally "
                            "drinks number of half-pint equivalents of alcoholic beverages drunk per day."}

    # Movement
    move = {'data': np.array([vect[:-1] for vect in data_dict['movement_libras']]).astype(float),
            'target': np.array([(int(vect[-1]) - 1) for vect in data_dict['movement_libras']]).astype(int),
            'description': "The data set contains 15 classes of 24 instances each. Each class references to a "
                           "hand movement type in LIBRAS (Portuguese name 'LÍngua BRAsileira de Sinais', "
                           "oficial brazilian signal language). Class labels 0-14"}

    # Parkinson's
    parkinsons = {
        'data': np.array([vect[1:len(vect) - 7] + vect[len(vect) - 6::] for vect in data_dict['parkinsons']]).astype(
            float),
        'target': np.array([vect[len(vect) - 7] for vect in data_dict['parkinsons']]).astype(int),
        'description': "This dataset is composed of a range of biomedical voice measurements from 31 people, "
                       "23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, "
                       "and each row corresponds one of 195 voice recording from these individuals. "
                       "The main aim of the data is to discriminate healthy people from those with PD. Labels are "
                       "0 for healthy and 1 for PD."}

    # Seeds
    seeds = {'data': np.array([vect[:-1] for vect in data_dict['seeds_dataset']]).astype(float),
             'target': np.array([(int(vect[-1]) - 1) for vect in data_dict['seeds_dataset']]).astype(int),
             'description': "Measurements of geometrical properties of kernels belonging "
                            "to three different varieties of "
                            "wheat (0-2). "
                            "A soft X-ray technique and GRAINS package were used to construct all seven, "
                            "real-valued attributes."}

    # Transfusion
    transfusion = {'data': np.array([vect[:-1] for vect in data_dict['transfusion']]).astype(float),
                   'target': np.array([vect[-1] for vect in data_dict['transfusion']]).astype(int),
                   'description': "To demonstrate the RFMTC marketing model (a modified version of RFM), "
                                  "this study adopted the donor database of Blood Transfusion Service Center "
                                  "in Hsin-Chu City in Taiwan. "
                                  "The center passes their blood transfusion service bus to "
                                  "one university in Hsin-Chu City to gather blood donated about every three months. "
                                  "To build a FRMTC model, we selected 748 donors at random from the donor database. "
                                  "These 748 donor data, each one included R (Recency - months since last donation), "
                                  "F (Frequency - total number of donation), M "
                                  "(Monetary - total blood donated in c.c.), "
                                  "T (Time - months since first donation), "
                                  "and a binary variable (class) representing whether he/she donated blood in March"
                                  " 2007 (1 stand for donating blood; 0 stands for not donating blood)."}

    # Wholesale
    wholesale = {'data': np.array([vect[2::] for vect in data_dict['Wholesale_customers_data']]).astype(float),
                 'target': np.array([(int(vect[1]) - 1) for vect in data_dict['Wholesale_customers_data']]).astype(int),
                 'description': "The data set refers to clients of a wholesale distributor. "
                                "It includes the annual spending in monetary units (m.u.) on diverse "
                                "product categories. "
                                "Clients are divided according to their region, i.e., Lisbon (0) N=77; Oporto (1) N=47;"
                                "Other Region (2) N=316."}

    # Yeast
    classlab_yeast = {'CYT': 0,
                      'ERL': 1,
                      'EXC': 2,
                      'ME1': 3,
                      'ME2': 4,
                      'ME3': 5,
                      'MIT': 6,
                      'NUC': 7,
                      'POX': 8,
                      'VAC': 9}
    yeast = {'data': np.array([vect[1:-1] for vect in data_dict['yeast']]).astype(float),
             'target': np.array([classlab_yeast[vect[-1]] for vect in data_dict['yeast']]).astype(int),
             'description': "Predicting the Cellular Localization Sites of Proteins. 10 locations labeled 0-9."}

    # Leaf
    classlab_leaf = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                     9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 21: 15,
                     22: 16, 23: 17, 24: 18, 25: 19, 26: 20, 27: 21, 28: 22,
                     29: 23, 30: 24, 31: 25, 32: 26, 33: 27, 34: 28, 35: 29}
    leaf = {'data': np.array([vect[2:-1] for vect in data_dict['leaf']]).astype(float),
            'target': np.array([classlab_leaf[int(vect[0]) - 1] for vect in data_dict['leaf']]).astype(int),
            'description': "This dataset consists in a collection of shape and texture features extracted from "
                           "digital images of leaf specimens originating from a total of 30 (0-29) "
                           "different plant species."}

    # Forest
    # 's': ('Sugi' forest), 'h': ('Hinoki' forest), 'd': ('Mixed deciduous' forest), 'o': ('Other' non-forest land)
    classlab_forest = {'s': 0,
                       'h': 1,
                       'd': 2,
                       'o': 3}
    forest = {'data': np.array(
        [vect[1::] for vect in data_dict['forest_tr']] + [vect[1::] for vect in data_dict['forest_ts']]).astype(float),
              'target': np.array(
                  [classlab_forest[vect[0]] for vect in data_dict['forest_tr']] + [classlab_forest[vect[0]] for vect in
                                                                                   data_dict['forest_ts']]).astype(int),
              'description': "Multi-temporal remote sensing data of a forested area in Japan. "
                             "It includes both training "
                             "and test datasets. "
                             "The goal is to map different forest types using spectral data. "
                             "Types: Sugi forest (0); "
                             "Hinoki forest (1); "
                             "Mixed deciduous (2); "
                             "Other (3)"}

    # Urban
    classlab_urban = {'asphalt': 0,
                      'building': 1,
                      'car': 2,
                      'concrete': 3,
                      'grass': 4,
                      'pool': 5,
                      'shadow': 6,
                      'soil': 7,
                      'tree': 8}
    urban = {'data': np.array(
        [vect[1::] for vect in data_dict['urban_tr']] + [vect[1::] for vect in data_dict['urban_ts']]).astype(float),
             'target': np.array(
                 [classlab_urban[vect[0]] for vect in data_dict['urban_tr']] + [classlab_urban[vect[0]] for vect in
                                                                                data_dict['urban_ts']]).astype(int),
             'description': "Contains merged training and testing data for classifying a high resolution "
                            "aerial image into 9 types of urban land cover (0-8). "
                            "Multi-scale spectral, size, shape, and texture information are "
                            "used for classification."}

    # Create dataset collection and dump it
    data_collection = UCI_datasets(biodeg, breastwi, climate, banknote, ecoli, glass, ionosphere, iris, liver, move,
                                   parkinsons, seeds, transfusion, wholesale, yeast, forest, urban, leaf)

    return data_collection
