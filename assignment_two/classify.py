from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def regression(df, features, target):

    '''Runs logistic regression

    Inputs:
        df: dataframe
        features: list
        target: string
    
    Return:
        model
    '''

    model = LogisticRegression()
    model.fit(df[features], df[target])

    print()
    print(features)
    print()
    print(model)
    print()

    return model


def evaluate_regression(model, df, features, target):

    '''Provides some evaluation metrics for the regression obtained above

    Inputs:
        model: model from regression function (above)
        df: dataframe
        features: list
        target: string
    
    Return:
        Prints some evaluation metrics
    '''

    #make predictions
    expected = df[target]
    predicted = model.predict(df[features])
    
    #summary of the precision, recall, F1 score for each class
    print('classification report: {}'.format(features))
    print(metrics.classification_report(expected, predicted)) 
    print()

    #confusion matrix: true negatives, false negatives, true positives, false positives
    print('confusion matrix: {}'.format(features))
    print(metrics.confusion_matrix(expected, predicted))
    print()

    #accuracy score (on training set)
    print('accuracy score (on training set): {}'.format(features))
    score = model.score(df[features], df[target])
    print(score)
    print()
    print()
    print()




