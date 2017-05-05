from sklearn import datasets
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
from scipy import optimize
import time
import seaborn as sns


def define_clfs_params(grid_size):
    '''adapted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py

    Return classifiers and parameter grid, based on input grid size
    
    SVM refusing to run for some reason?
    
    '''


    classifiers = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        #'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
		'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=5, max_samples=0.65, max_features=1)
            }
    
    mini_grid = { 
        'RF': {'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
        'LR': { 'penalty': ['l1','l2'], 'C': [1,10]},
        #'SVM': {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'GB': {'n_estimators': [10,100], 'learning_rate' : [0.1,0.5],'subsample' : [0.3,1.0], 'max_depth': [5,50]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [10,50], 'max_features': ['sqrt','log2'],'min_samples_split': [3,10]},
        'KNN': {'n_neighbors': [5,25],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,1000]},
        'BAG': {'n_estimators': [5,20], 'max_samples':[0.25,0.75]}
           }
        
    practice_grid = { 
        'RF': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
        'LR': { 'penalty': ['l1'], 'C': [0.01]},
        #'SVM' :{'C' :[0.01],'kernel':['linear']},
        'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
        'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},        
        'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
        'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
		'BAG': {'n_estimators': [5]}
            }
    
    if grid_size == 'practice':
        return classifiers, practice_grid
    
    else:
        return classifiers, mini_grid



def test_train(df, features, y):
    '''
    Splits the data into training and testing data

    Inputs:
        df: dataframe
        features: features of the dataframe that are being used (list)
        y: target/outcome variable (string)
    
    Return:
        X_train: features from training set
        X_test: features from test set
        y_train: targets from training set
        y_test: targets from test set

    '''

    X = df[features]
    y = df[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    return X_train, X_test, y_train, y_test


def generate_binary_at_k(y_scores, k):
    '''adapted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py'''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def eval_metrics(y_true, y_scores, k):
    '''adapted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    
    Returns dictionary with precision, recall, accuracy, and f1 scores

    accuracy = correct / total
    precision = true positive / predicted positive
    recall = true positive / true
    f1 = 2 (precision * recall) / precision + recall

    '''

    scores = {}
    preds_at_k = generate_binary_at_k(y_scores, k)

    precision = precision_score(y_true, preds_at_k)
    scores['precision'] = precision

    recall = recall_score(y_true, preds_at_k)
    scores['recall'] = recall

    accuracy = accuracy_score(y_true, preds_at_k)
    scores['accuracy'] = accuracy

    f1 = f1_score(y_true, preds_at_k)
    scores['f1'] = f1

    return scores


def model_loop(df, features, y, grid_size):
    '''adapted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    
    Loops through models and parameters; prints and returns dataframe of evaluation metrics
    
    '''
    
    #initialize dataframe to store results
    results_df =  pd.DataFrame(columns=('model_type', 'parameters', 'auc-roc', 
        'prec_at_5', 'rec_at_5', 'acc_at_5', 'f1_at_5', 
        'prec_at_10', 'rec_at_10', 'acc_at_10', 'f1_at_10',
        'prec_at_20', 'rec_at_20' , 'acc_at_20', 'f1_at_20', 
        'time'))
    
    #get training/testing sets
    X_train, X_test, y_train, y_test = test_train(df, features, y)

    #get info about classifiers and parameters
    classifiers, grid = define_clfs_params(grid_size)

    #loop through models
    for model, classifier in classifiers.items():
        print (model)
        parameters = grid[model]

        #loop through parameters
        for param in ParameterGrid(parameters):
            print(param)
            try:
                classifier.set_params(**param)

                #time the process
                start_time = time.time()
                
                #fit model and make predictions
                y_pred_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:,1]

                #sort predictions
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

                #get the total time
                end_time = time.time()
                total_time = end_time - start_time

                #get evaluation metrics
                scores5 = eval_metrics(y_test_sorted, y_pred_probs_sorted, 5.0)
                scores10 = eval_metrics(y_test_sorted, y_pred_probs_sorted, 10.0)
                scores20 = eval_metrics(y_test_sorted, y_pred_probs_sorted, 20.0)

                #fill in results dataframe
                results_df.loc[len(results_df)] = \
                    [model, param, roc_auc_score(y_test, y_pred_probs),
                        scores5['precision'], scores5['recall'], scores5['accuracy'], scores5['f1'],
                        scores10['precision'], scores10['recall'], scores10['accuracy'], scores10['f1'],
                        scores20['precision'], scores20['recall'], scores20['accuracy'], scores20['f1'],
                        total_time]

            except IndexError:
                print('index error')
                continue

    print(results_df)
    return results_df

def evaluate_results(results_df):

    '''prints highest evaluation metrics'''
    
    index = results_df['auc-roc'].argmax()
    print('highest auc-roc: {} ({})'.format(results_df['auc-roc'].max(), results_df.ix[index, 'model_type']))

    index = results_df['prec_at_5'].argmax()
    print('highest precision at k=5: {} ({})'.format(results_df['prec_at_5'].max(), results_df.ix[index, 'model_type']))

    index = results_df['rec_at_5'].argmax()
    print('highest recall at k=5: {} ({})'.format(results_df['rec_at_5'].max(), results_df.ix[index, 'model_type']))

    index = results_df['acc_at_5'].argmax()
    print('highest accuracy at k=5: {} ({})'.format(results_df['acc_at_5'].max(), results_df.ix[index, 'model_type']))

    index = results_df['f1_at_5'].argmax()
    print('highest f1 at k=5: {} ({})'.format(results_df['f1_at_5'].max(), results_df.ix[index, 'model_type']))

    index = results_df['prec_at_10'].argmax()
    print('highest precision at k=10: {} ({})'.format(results_df['prec_at_10'].max(), results_df.ix[index, 'model_type']))

    index = results_df['auc-roc'].argmax()
    print('highest recall at k=10: {} ({})'.format(results_df['rec_at_10'].max(), results_df.ix[index, 'model_type']))

    index = results_df['rec_at_10'].argmax()
    print('highest accuracy at k=10: {} ({})'.format(results_df['acc_at_10'].max(), results_df.ix[index, 'model_type']))

    index = results_df['f1_at_10'].argmax()
    print('highest f1 at k=10: {} ({})'.format(results_df['f1_at_10'].max(), results_df.ix[index, 'model_type']))

    index = results_df['prec_at_20'].argmax()
    print('highest precision at k=20: {} ({})'.format(results_df['prec_at_20'].max(), results_df.ix[index, 'model_type']))

    index = results_df['rec_at_20'].argmax()
    print('highest recall at k=20: {} ({})'.format(results_df['rec_at_20'].max(), results_df.ix[index, 'model_type']))

    index = results_df['acc_at_20'].argmax()
    print('highest accuracy at k=20: {} ({})'.format(results_df['acc_at_20'].max(), results_df.ix[index, 'model_type']))

    index = results_df['f1_at_20'].argmax()
    print('highest f1 at k=20: {} ({})'.format(results_df['f1_at_20'].max(), results_df.ix[index, 'model_type']))