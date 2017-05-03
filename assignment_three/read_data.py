import pandas as pd

def read_data(file_type, file_path):
    '''Reads data into a pandas dataframe

    Input: file_type (string), file_path (string)
    Return: dataframe

    In the future: add functionality for other filetypes
    '''

    if file_type == 'csv':
        df = pd.read_csv(file_path)
    
    return df