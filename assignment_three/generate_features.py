import pandas as pd

def bin_data(df, column, num_percentiles=4):

    '''Discretize a continuous variable.

    Inputs:
        df: dataframe on which function is to be applied
        column: column on which function is to be applied
        num_percentiles: number of percentiles into which values should be broken
    
    Return:
        df: returns original dataframe, now with an additional column including
            the category names associated with each observation
        
        Also prints the range of values for each bin
    
    In the future: create another function that creates bins with equal ranges,
                    rather than basing ranges on percentages
    
    '''

    value = 1/num_percentiles
    base_value = 1/num_percentiles

    bin_min = df.min()[column]
    bins = [bin_min]

    while value < 1:

        column_bin = df.quantile(q=value)[column]


        if value < 1:
            bins.append(column_bin)

        value += base_value

    
    if df[column].max() not in bins:
        bin_max = df.max()[column]
        bins.append(bin_max)

    bin_names = []

    for i in range(len(bins) - 1):
        bin_names.append(int('{}'.format(i)))

    df['{}_categories'.format(column)] = pd.cut(df[column], bins, labels=bin_names)

    bin_ranges = []
    for i, val in enumerate(bin_names):
        if i + 1 <= len(bin_names):
            low = bins[i]
            high = bins[i+1]
            bin_ranges.append('{}: {} - {}'.format(bin_names[i], low, high))
        
    for index, row in df.iterrows():
        if df.ix[index, column] == 0:
            df.set_value(index, '{}_categories'.format(column), 0.0)

    print('bins: {}'.format(column))
    print(bin_ranges)
    
    return df


def create_dummies(df, column):
    '''Create new columns with dummy variables based on the given column.

    Inputs:
        df: dataframe
        column: dataframe column from which dummies should be created
    
    Return:
        new dataframe with additional columns (original column is removed)
    '''

    new_df = pd.get_dummies(df, columns=[column])

    return new_df










