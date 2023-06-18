import re #To remove non-alphabetic characters
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from utils.constants import RAW_DATA_SET

count_progress=1
len_df=0
def remove_stopwords(review):
    global count_progress
    print("Removing Stopwords: {}%".format((count_progress/len_df)*100))
    rev_tokens=word_tokenize(review)
    rev_tokens_without_sw = [word for word in rev_tokens if word not in stopwords]
    rev_without_sw = ""
    for token in rev_tokens_without_sw:
        rev_without_sw += token+" "
    count_progress+=1
    return rev_without_sw

def stem(review):
    global count_progress
    print("Stemming: {}%".format((count_progress/len_df)*100))
    stemmed_rev=""
    clean_rev_tokens = word_tokenize(review)
    porter=PorterStemmer()
    for token in clean_rev_tokens:
        stemmed_rev+=porter.stem(token)+" "
    stemmed_rev=stemmed_rev.strip()
    count_progress+=1
    return stemmed_rev

def build_datasets():
    global len_df, count_progress
    df = pd.read_csv(f'./datasets/{RAW_DATA_SET}')
    print(df['sentiment'])
    df['sentiment'] = df['sentiment'].apply(lambda row: 1 if row=='positive' else 0)
    len_df=df.shape[0]
    print(df['sentiment'])
    print(df['review'])
    df['review']=df.apply(lambda row: re.sub(r'(<[\w\s]*/?>)',"",row['review']), axis=1)
    #df['review']=df.apply(lambda row: re.sub(r'[^a-zA-Z0-9\s]+', '', row['review']), axis=1)
    df['review'] = df.apply(lambda row: re.sub(r'[^a-zA-Z\s]+', '', row['review']), axis=1)
    print(df['review'])
    df['review']=df.apply(lambda row: remove_stopwords(row['review']), axis=1)
    count_progress=1
    print(df['review'])
    df['review']=df.apply(lambda row: stem(row['review']), axis=1)
    count_progress=1
    print(df['review'])
    df.to_csv('clean_rev_movies.csv', index=False)
#build_datasets()