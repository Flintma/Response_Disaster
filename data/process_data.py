import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''Load , merge messages , categories datasets
    Input: messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    outputs:
    df: dataframe. Dataframe containing merged content of messages , categories datasets.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    df_temp_id = df['id']
    return df, df_temp_id


def clean_data(df, df_temp_id):
    """Clean dataframe by removing duplicates , converting categories from strings 
    to binary values.
    
    Args:
    df: dataframe. Dataframe containing merged content of messages , categories datasets.
       
    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    
    
    categories =  df['categories'].str.split(';', expand=True).add_prefix('categories_')
    messages = df[['message', 'genre', 'id']]
    row = categories.iloc[0]
    category_colnames = list()
    for x in row:
        #print(x[0:-2])
        category_colnames.append(x[0:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    categories['id'] = df['id']

    df = pd.merge(messages, categories)
    # check number of duplicates
    print(df.duplicated().sum())
    # drop duplicates
    df.drop_duplicates(inplace = True)
    # check number of duplicates
    print(df.duplicated().sum())
    return df
    

def save_data(df, database_filename):
    """Save into  SQLite database.
    
    inputs:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    outputs:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    '''Main program to run the ETL
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, df_temp_index = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, df_temp_index)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()