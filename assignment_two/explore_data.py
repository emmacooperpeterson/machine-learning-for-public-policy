import pandas as pd
import matplotlib.pyplot as plt

def overview(df):
    rows, columns = df.shape

    print ('Rows: {}'.format(rows), 'Columns: {}'.format(columns))
    print()
    print ('Column names: ', list(df.columns))
    print()
    print('Data types')
    print(df.dtypes)

def view_nulls(df):

    '''Source: https://github.com/yhat/DataGotham2013/blob/master/notebooks/3%20-%20Importing%20Data.ipynb'''
    
    df2 = pd.melt(df)
    nulls = df2.value.isnull()
    
    print (pd.crosstab(df2.variable, nulls))


def group_by(df, column):
    '''find the number of observations for each value in the given column'''

    counts = df[column].value_counts()

    print (counts)


def describe_data(df):
    '''get a table describing the min/max/mean/median of each column.
    may or may not make sense for every column, but allows you to quickly
    review data'''
    
    results = df.describe()

    print (results)


def get_plot(df, column, normalize=False, plot_type='hist'):

    '''plot types: hist, box, pie'''

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








