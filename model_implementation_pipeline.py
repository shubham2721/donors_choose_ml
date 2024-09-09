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
import seaborn as sns

import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# function that will split the data into Train, CV and Test

def train_cv_test_split(data, target_variable):
    from sklearn.model_selection import train_test_split
    y = data[target_variable].values # Target Variable
    X = data.drop([target_variable], axis=1) # Independent Variables
    # Splitting the data into train test and CV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, stratify = y)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size = 0.33, stratify = y_train)
    return X_train, X_cv, X_test, y_train, y_cv, y_test

# #calling above function to transform the data
# X_train_num, X_train_cat, X_train_des = feature_transform(X_train)
# X_cv_num, X_cv_cat, X_cv_des = feature_transform(X_cv)
# X_test_num, X_test_cat, X_test_des = feature_transform(X_test)

# # storing the features value together in CSR
# X_tr = hstack((X_train_num, X_train_cat, X_train_des)).tocsr()
# X_cr = hstack((X_cv_num, X_cv_cat, X_cv_des)).tocsr()
# X_te = hstack((X_test_num, X_test_cat, X_test_des)).tocsr()

# model training

def evaluate_model(estimator, cv, X_train_data, y_train_data, param):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt

    clf = RandomizedSearchCV(estimator, param_distributions = param, cv = cv, scoring= 'roc_auc', return_train_score=True)
    clf.fit(X_train_data, y_train_data)

    results = pd.DataFrame.from_dict(clf.cv_results_)
    if 'param_n_neighbors' in results.columns:
        results = results.sort_values(['param_n_neighbors'])
    elif 'param_C' in results.columns:
        results = results.sort_values(['param_C'])
    else:
        # Plotting Heat map to see the best hyper parameter both in Train and Test data
        pivot_train = results.pivot(index = 'param_min_samples_split', columns = 'param_max_depth', values = 'mean_train_score')
        pivot_test = results.pivot(index = 'param_min_samples_split', columns = 'param_max_depth', values = 'mean_test_score')
        # Plotting heatmap-
        fig, ax = plt.subplots(1,2, figsize=(20,12))
        sns.heatmap(pivot_train, annot = True, ax = ax[0])
        ax[0].title.set_text('Train ROC score')
        sns.heatmap(pivot_test, annot = True, ax = ax[1])
        ax[1].title.set_text('Test ROC score')
        return fig.show()

    train_auc = results['mean_train_score']
    cv_auc = results['mean_test_score']
    if 'param_n_neighbors' in results.columns:
        k = results['param_n_neighbors']
    elif 'param_C' in results.columns:
        k = results['param_C']
    
    # Plotting Error Graph for both log-reg and KNN
    plt.plot(k, train_auc, label = 'Train AUC')
    plt.plot(k, cv_auc, label = 'CV AUC')

    plt.scatter(k, train_auc, label = 'Train AUC')
    plt.scatter(k, cv_auc, label = 'CV AUC')

    plt.legend()
    plt.xlabel("hyperparameter")
    plt.ylabel("AUC")
    if 'param_n_neighbors' in results.columns:
        pass
    else:
        plt.xscale("log")
    plt.title("Hyper parameter Vs AUC plot")
    plt.grid()
    return plt.show()

def model_imp(best_k, best_c, max_depth, min_s_s, X_tr, y_train):

    neigh = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    neigh.fit(X_tr, y_train)
    pickle.dump(neigh, open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/KNN.pkl', 'wb'))

    log = LogisticRegression(C = best_c, max_iter= 1000, n_jobs = -1)
    log.fit(X_tr, y_train)
    pickle.dump(log, open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/LogisticRegression.pkl', 'wb'))

    dt = DecisionTreeClassifier(random_state=0, max_depth = max_depth, min_samples_split = min_s_s)
    dt.fit(X_tr, y_train)
    pickle.dump(dt, open('/Users/shubhamshivendra/workspace/Project/Donors Choose/Data Sets/DecisionTree.pkl', 'wb'))

    return neigh, log, dt


# this function will take parameter estimator and data and will yield probability score in batches
def batch_pred(clf, data):
    y_data_pred = []
    data_loop = data.shape[0] - data.shape[0] % 1000
    for i in range(0, data_loop, 1000):
        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])
    if data.shape[0]%1000 !=0:
        y_data_pred.extend(clf.predict_proba(data[data_loop:])[:,1])
    return y_data_pred

def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
