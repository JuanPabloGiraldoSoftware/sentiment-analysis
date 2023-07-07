import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.constants import CLEAN_DATA_SET, RAW_DATA_SET
from utils.visualization import print_shape, print_average_length, print_bal_class


def read_dataset(path):
    df_clean = pd.read_csv(f'./datasets/{path}')
    return df_clean

def corpus_statics():
    #Corpus Statistics
    df_raw = read_dataset(RAW_DATA_SET)
    df_clean =  read_dataset (CLEAN_DATA_SET)

    print_bal_class('Balance',df_raw["sentiment"])
    print_average_length('RAW DATASET AVLEN:', df_raw['review'])
    print_average_length('CLEAN DATASET AVLEN:',df_clean['review'])
    

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

    # Convertir las matrices dispersas en matrices densas
    x_train_bow = x_train_bow.toarray()
    x_test_bow = x_test_bow.toarray()

    # Remodelar las matrices para que tengan la forma adecuada
    x_train_vec = np.reshape(x_train_bow, (x_train_bow.shape[0], x_train_bow.shape[1], 1))
    x_test_vec = np.reshape(x_test_bow, (x_test_bow.shape[0], x_test_bow.shape[1], 1))

    return x_train_vec, y_train, x_test_vec, y_test