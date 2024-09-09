import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import model_implementation_pipeline as mip
import features_transformation_pipeline as ftp
from scipy.sparse import hstack

import pickle


def feature_transform(data):
    scaler = pickle.load(open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/scaler.pkl', 'rb'))
    num = scaler.transform(data[scaler.feature_names_in_])
    ohe = pickle.load(open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/ohe.pkl', 'rb'))
    cat = ohe.transform(data[ohe.feature_names_in_])
    essay_vectorize = ftp.avg_word_2vec(data)
    data_new = hstack((num, cat, essay_vectorize)).tocsr()
    return data_new

def predict_score_inf(project_data, resource_data):
    new_data = ftp.preprocess_data(project_data, resource_data)
    df = feature_transform(new_data)
    log = pickle.load(open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/LogisticRegression.pkl', 'rb'))
    y_train_pred = mip.batch_pred(log, df)
    final = mip.predict_with_best_t(y_train_pred, 0.8347440976017594)
    return final, y_train_pred