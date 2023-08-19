import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.constants import CLEAN_DATA_SET, RAW_DATA_SET
from utils.visualization import print_shape, print_average_word_length,print_average_char_length, print_bal_class


def read_dataset(path):
    df_clean = pd.read_csv(f'./datasets/{path}')
    return df_clean

def corpus_statics():
    #Corpus Statistics
    df_raw = read_dataset(RAW_DATA_SET)
    df_clean =  read_dataset (CLEAN_DATA_SET)

    print_bal_class('Balance',df_raw["sentiment"])
    print_average_word_length('RAW DATASET AV. WORD LEN:', df_raw['review'])
    print_average_word_length('CLEAN DATASET AV. WORD LEN:',df_clean['review'])
    print_average_char_length('RAW DATASET AV. CHAR LEN:', df_raw['review'])
    print_average_char_length('CLEAN DATASET AV. CHAR LEN:',df_clean['review'])
    

def default_tfidf_vector(max_features=None):
    corpus_statics()
    df = read_dataset(CLEAN_DATA_SET)
    print(df.shape)
    X = df['review']
    Y = df['sentiment']
    vectorizer = TfidfVectorizer( max_features=max_features, max_df=0.5)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.30)
    print_shape(x_train, x_test)
    #vocabulary and idf learning > returns document term matrix
    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)
    print_shape(x_train_bow, x_test_bow)

    return x_train_bow, y_train, x_test_bow, y_test