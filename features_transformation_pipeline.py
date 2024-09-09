# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
# from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import tqdm

import re
#Tutorial about Python regular expressions: https://pymotw.com/2/re/

import pickle

# import plotly.express as px
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in text_data:
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text

def preprocess_data(data_1, data_2):
    price_data = data_2.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
    project_data = pd.merge(data_1, price_data, on='id', how='left')

    # https://stackoverflow.com/questions/36383821/pandas-dataframe-apply-function-to-column-strings-based-on-other-column-value
    project_data['project_grade_category'] = project_data['project_grade_category'].str.replace(' ','_')
    project_data['project_grade_category'] = project_data['project_grade_category'].str.replace('-','_')
    project_data['project_grade_category'] = project_data['project_grade_category'].str.lower()

    project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(' The ','')
    project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(' ','')
    project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace('&','_')
    project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(',','_')
    project_data['project_subject_categories'] = project_data['project_subject_categories'].str.lower()

    project_data['teacher_prefix']=project_data['teacher_prefix'].fillna('Mrs.')
    project_data['teacher_prefix'] = project_data['teacher_prefix'].str.replace('.','')
    project_data['teacher_prefix'] = project_data['teacher_prefix'].str.lower()

    project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(' The ','')
    project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(' ','')
    project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace('&','_')
    project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(',','_')
    project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.lower()

    project_data['school_state'] = project_data['school_state'].str.lower()

    preprocessed_titles = preprocess_text(project_data['project_title'].values)

    project_data["essay"] = project_data["project_essay_1"].map(str) +\
                        project_data["project_essay_2"].map(str) + \
                        project_data["project_essay_3"].map(str) + \
                        project_data["project_essay_4"].map(str)
    
    preprocessed_essays = preprocess_text(project_data['essay'].values)

    project_data['essay'] = preprocessed_essays
    project_data['project_title'] = preprocessed_titles
    project_data['clean_categories'] = project_data['project_subject_categories']
    project_data['clean_subcategories'] = project_data['project_subject_subcategories']

    if 'project_is_approved' in project_data.columns:
        project_data = project_data[['school_state',
                    'teacher_prefix',
                    'project_grade_category',
                    'teacher_number_of_previously_posted_projects',
                    'project_is_approved',
                    'clean_categories',
                    'clean_subcategories',
                    'essay',
                    'price']]
    else:
        project_data = project_data[['school_state',
                    'teacher_prefix',
                    'project_grade_category',
                    'teacher_number_of_previously_posted_projects',
                    'clean_categories',
                    'clean_subcategories',
                    'essay',
                    'price']]
    return project_data


def feature_transformation_fit(train_data): # only training data will be passed
    '''THis function will first seperate all the different data types and will transform the data 
    1. If the feature's datatype is numeric then standard scaler will be performed
    2. If it's categorical then one hot encoding will be performed 
    3. And if it's descriptive then vectorization using glove file
    '''
    # dropping descriptive data
    train_data = train_data.drop('essay', axis = 1)

    # Numeric data (including integers and floats)
    numeric_data = train_data.select_dtypes(include=[np.number]).columns

    # Categorical data (including strings and other non-numeric types)
    categorical_data = train_data.select_dtypes(exclude=[np.number]).columns

    print(numeric_data, "numerical columns")
    print(categorical_data, "categorical columns")
    scaler = StandardScaler()
    scaler.fit(train_data[numeric_data])
    pickle.dump(scaler, open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/scaler.pkl', 'wb'))

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(train_data[categorical_data])
    pickle.dump(ohe, open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/ohe.pkl', 'wb'))
    return scaler, ohe

def avg_word_2vec(data):
     # this will be the descriptive value
    with open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/glove_vectors', 'rb') as f:
        model = pickle.load(f)
        # storing the words in the variable
        glove_words = set(model.keys())
    # Average Word2Vec for Train data
    avg_w2v_vectors  = []
    for sentence in data['essay'].values:
        vector = np.zeros(300) # as the model is trained on 300D we need to initialize each word in sentence with 300D
        cnt_words = 0
        for word in sentence.split():
            if word in glove_words:
                vector += model[word]
                cnt_words += 1
        if cnt_words != 0:
            vector /= cnt_words
        avg_w2v_vectors.append(vector)
    return avg_w2v_vectors