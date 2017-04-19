import pandas as pd

def read_data(file_type, file_path):

    if file_type == 'csv':
        df = pd.read_csv(file_path)
    
    return df