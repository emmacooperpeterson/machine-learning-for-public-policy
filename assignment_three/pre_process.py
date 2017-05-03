import pandas as pd
import re

def fill_missing(df, column, filler='mean'):
    '''fill in NaNs in the dataframe using the given filler value

    Inputs: dataframe, column name, filler (string) – default is mean
    Return: dataframe with missing values filled in for given column

    CHANGES FOR ASSIGNMENT THREE: updated so that you can fill missing values for
    a specific column, rather than the entire dataframe at once. added ability to filler
    with mode or median instead of mean for categorical variables

    '''
    if filler == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    
    if filler == 'mode':
        df[column].fillna(df[column].mode(), inplace=True)
    
    if filler == 'median':
        df[column].fillna(df[column].median(), inplace=True)


    return df


def camel_to_snake(column):
    '''converts a string that is camelCase into snake_case

    CHANGES FOR ASSIGNMENT THREE: fixed inaccuracy in header

    Input: column name
    Return: string (camel_to_snake version of column name)

    Source: https://github.com/yhat/DataGotham2013/blob/master/notebooks/3%20-%20Importing%20Data.ipynb
    '''

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()