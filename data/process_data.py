import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    Loads category and messsage data
    
    INPUTS
    categories: dataset containing classification of messages
    messages: dataset of actual text of messages
    
    OUPUT
    Merged dataframe of messages and their corresponding classified categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, left_on='id', right_on='id')
    
    return df


def clean_data(df):
    
    '''
    Cleans dataset for model training
    
    INPUT
    df: dataset of merged messages and their categories
    
    OUPTU
    df: cleaned dataset
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: str(x)[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
        
    # concatenate the original dataframe with the new `categories` dataframe
    frames = [df, categories]
    df = pd.concat(frames, axis=1)
    
    # check number of duplicates
    df.duplicated(keep='first').sum()
    
    # drop duplicates
    df = df.drop_duplicates()
    
    #Creates dummy variables for 'genre' category
#     genre_table = pd.get_dummies(df['genre'])
#     genre_table = genre_table.astype(int)
#     df = pd.concat([df, genre_table], axis=1)
    
    #Drops original 'genre' column
#     df = df.drop('genre', axis=1)
    
    return df
    
def save_data(df, database_filename):
    
    '''
    Saves dataset to SQlite database
    
    INPUT
    df: dataset of merged categories and messages
    database_filename: filepath for desired storage of SQlite table
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print (len(sys.argv))
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()