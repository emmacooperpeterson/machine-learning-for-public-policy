import pandas as pd
import matplotlib.pyplot as plt

def overview(df):
    '''Get some basic information about the dataframe

    Input: dataframe
    Return: prints number of rows/columns, column names, and data types for each column
    '''

    rows, columns = df.shape
    print ('Rows: {}'.format(rows), 'Columns: {}'.format(columns))
    print()
    print ('Column names: ', list(df.columns))
    print()
    print('Data types')
    print(df.dtypes)

def view_nulls(df):
    '''View how many nulls are present in each column

    Input: dataframe
    Return: prints table of nulls per column

    Source: https://github.com/yhat/DataGotham2013/blob/master/notebooks/3%20-%20Importing%20Data.ipynb
    '''
    
    df2 = pd.melt(df)
    nulls = df2.value.isnull()
    print (pd.crosstab(df2.variable, nulls))


def group_by(df, column):
    '''Find the number of observations for each value in the given column
    
    Inputs: dataframe, column on which to group
    Return: prints the counts for each unique value in that column
    '''

    counts = df[column].value_counts()
    print (counts)


def describe_data(df):
    '''Get a table with descriptive statistics for each column.

    Input: dataframe
    Return: prints descriptive statistics

    The stats may or may not make sense for every column, 
    but this allows you to quickly review data
    
    '''
    
    results = df.describe()
    print (results)


def get_plot(df, column, normalize=False, plot_type='hist'):

    '''View and save charts based on a dataframe column

    Inputs: dataframe, column, True/False for histogram normalization, and plot_type (string)
    Return: saves the chart as png

    Currently supported plot types: hist, box, pie
    '''

    if plot_type == 'hist': 
        df[column].plot(plot_type, normed=normalize, title=column)
    
    elif plot_type == 'box':
        df[column].plot(plot_type, title=column)
    
    elif plot_type == 'pie':
        l = None
        
        if len(set(df[column])) == 2:
            l = ['No', 'Yes']

        df[column].value_counts().plot.pie(figsize=(6, 6), title=column, legend=True, labels=l)
    
    print('chart saved')
    path = '{}_{}.png'.format(column, plot_type)
    plt.savefig(path)
    plt.show()








