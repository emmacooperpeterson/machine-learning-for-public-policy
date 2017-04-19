from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def regression(df, features, target):

    #fit model to data
    model = LogisticRegression()
    model.fit(df[features], df[target])

    print()
    print(features)
    print()
    print(model)
    print()

    return model

def evaluate_regression(model, df, features, target):

    #make predictions
    expected = df[target]
    predicted = model.predict(df[features])
    
    #Text summary of the precision, recall, F1 score for each class
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




