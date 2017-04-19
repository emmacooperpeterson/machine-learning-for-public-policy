import pandas as pd
import re

def fill_missing(df, filler='mean'):
    '''fill in NaNs in the dataframe using the given filler value

    Inputs: dataframe, filler (string) – default is mean
    Return: dataframe with missing values filled in

    In the future: make this column specific and add functionality for other
    types of filler values

    '''

    df = df.fillna(df.mean())
    return df


def camel_to_snake(column):
    '''converts a string that is camelCase into snake_case

    Input: column name
    Return: dataframe with updated column name

    Source: https://github.com/yhat/DataGotham2013/blob/master/notebooks/3%20-%20Importing%20Data.ipynb
    '''

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()