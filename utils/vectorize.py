import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.constants import CLEAN_DATA_SET
from utils.visualization import print_shape


def read_clean_dataset():
    df_clean = pd.read_csv(f'./datasets/{CLEAN_DATA_SET}')
    return df_clean

def default_tfidf_vector(max_features=None):
    df = read_clean_dataset()
    print(df.shape)
    X = df['review']
    Y = df['sentiment']
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.30)
    print_shape(x_train, x_test)
    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)
    print_shape(x_train_bow, x_test_bow)
    return x_train_bow, y_train, x_test_bow, y_test