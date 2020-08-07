import sys
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    
    """load the files"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    
    return df

def clean_data(df):
    """Cleans the data"""
    category_col = 'categories'
    categories = df[category_col].str.split(';', expand=True)

    ##get the first row and extract the column nemes
    category_colnames = list(categories.iloc[0].str.split('-').str[0])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1 )
    #check number of duplicates
    df[df.duplicated()].shape[0]
    df = df.drop_duplicates()
    
    return df
    
def save_data(df, database_filename):
    """saves the dataframe in a SQLite database"""
   engine = create_engine('sqlite:///'+database_filename)
   df.to_sql('Message_Categories', engine, index=False)  


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()