#Author: Juan Pablo Giraldo M.#

###################################################
#          Raw datasets sources                   #
###################################################
#movies1.csv Link: https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
#movies2.csv Link: https://www.kaggle.com/code/yasserh/imdb-movie-rating-sentiment-analysis/data


###################################################
#          libraries and external functions       #
###################################################
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re #To remove non-alphabetic characters
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
count_progress=1
len_df=0
def remove_stopwords(review):
    global count_progress
    print("Removing Stopwords: {}%".format((count_progress/len_df)*100))
    rev_tokens=word_tokenize(review)
    rev_tokens_without_sw = [word for word in rev_tokens if not word in stopwords]
    rev_without_sw = ""
    for token in rev_tokens_without_sw:
        rev_without_sw += token+" "
    #print(rev_without_sw)
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
    #print(stemmed_rev)
    count_progress+=1
    return stemmed_rev

def bag_of_words(review):
    global count_progress
    print("Vectorizing: {}%".format((count_progress/len_df)*100))
    stemmed_rev_tokens=word_tokenize(review)
    vocab=[]
    for word in stemmed_rev_tokens:
        if word not in vocab:
            vocab.append(word)
    #print(vocab)
    index_word={}
    i = 0   
    for word in vocab:
        index_word[word]=i
        i+=1
    count_dict = defaultdict(int)
    vec = [0 for _ in range(len(vocab))]
    for item in stemmed_rev_tokens:
        count_dict[item] += 1
    for key,item in count_dict.items():
        vec[index_word[key]] = item
    #print(vec)
    count_progress+=1
    return vec

def build_datasets():
    global len_df, count_progress
    df = pd.read_csv("concat_movies.csv")
    print(df['sentiment'])
    df['sentiment'] = df['sentiment'].apply(lambda row: 1 if row=='positive' else 0)
    len_df=df.shape[0]
    print(df['sentiment'])
    print(df['review'])
    df['review']=df.apply(lambda row: re.sub(r'(<[\w\s]*/?>)',"",row['review']), axis=1)
    df['review']=df.apply(lambda row: re.sub(r'[^a-zA-Z0-9\s]+', '', row['review']), axis=1)
    print(df['review'])
    df['review']=df.apply(lambda row: remove_stopwords(row['review']), axis=1)
    count_progress=1
    print(df['review'])
    df['review']=df.apply(lambda row: stem(row['review']), axis=1)
    count_progress=1
    print(df['review'])
    df.to_csv('clean_rev_movies.csv', index=False)
    df['review']=df.apply(lambda row: bag_of_words(row['review']), axis=1)
    count_progress=1
    df.to_csv('bow_rev_movies.csv', index=False)
    print(df['review'])
#build_datasets()

def print_shape(a,b):
    """
    Function that prints the shape of the numpy arrays passed as arguments
    """
    print("Size of Training Samples")
    print("="*30)
    print(a.shape)
    print("Size of Testing Samples")
    print("="*30)
    print(b.shape)

def training_model():
    df = pd.read_csv("clean_rev_movies.csv")
    df = df.loc[:40000]
    X=df['review']
    Y=df['sentiment']
    print(X)
    print(Y)
    print("###################################################")
    vectorizer = CountVectorizer()
    x_train, x_test, y_train, y_test = train_test_split(X,Y,stratify=Y, test_size=0.33)
    print_shape(x_train,x_test)
    x_train_bow=vectorizer.fit_transform(x_train)
    x_test_bow=vectorizer.transform(x_test)
    print_shape(x_train_bow,x_test_bow)
    grid_params = { 'n_neighbors' : [40,50,60,70,80,90],
               'metric' : ['manhattan']}
    knn=KNeighborsClassifier()
    clf = RandomizedSearchCV(knn, grid_params, random_state=0,n_jobs=-1,verbose=1)
    clf.fit(x_train_bow,y_train)
    print(clf.best_params_)
    print(clf.best_score_)
    print(clf.cv_results_)
    train_fpr,train_tpr,thresholds=roc_curve(y_train,clf.predict_proba(x_train_bow)[:,1])
    test_fpr,test_tpr,thresholds=roc_curve(y_test,clf.predict_proba(x_test_bow)[:,1])
    plt.plot(train_fpr,train_tpr,label="Training Accuracy="+str(round(auc(train_fpr, train_tpr),2)))
    plt.plot(test_fpr,test_tpr,label="Testing Accuracy ="+str(round(auc(test_fpr, test_tpr),2)))
    plt.legend()
    plt.xlabel("Thresholds")
    plt.ylabel("ACCURACY")
    plt.title("Training and Testing ROC Curves")
    plt.show()
    

training_model()



def single_test():
    rev=df2.loc[2].review
    print(rev)
    #Removing html <br /> tag

    compile_html_tag=re.compile('(\s*)<br />(\s*)')
    no_html_rev= compile_html_tag.sub(' ',rev)
    print("===============================================================================")
    print(no_html_rev)
    n_rev= re.sub(r'[^a-zA-Z\s]', '', no_html_rev)
    print("===============================================================================")
    print(n_rev)
    #Remving all stopwords
    rev_tokens=word_tokenize(n_rev)
    rev_tokens_without_sw = [word for word in rev_tokens if not word in stopwords.words()]
    print("===============================================================================")
    print(rev_tokens_without_sw)
    rev_without_sw = ""
    for token in rev_tokens_without_sw:
        rev_without_sw += token+" "
    print("===============================================================================")
    rev_without_sw=rev_without_sw.strip()
    print(rev_without_sw)
    #Stemming
    stemmed_rev=""
    clean_rev_tokens = word_tokenize(rev_without_sw)
    porter=PorterStemmer()
    for token in clean_rev_tokens:
        stemmed_rev+=porter.stem(token)+" "
    stemmed_rev=stemmed_rev.strip()
    print("===============================================================================")
    print(stemmed_rev)
    #Bag of words
    stemmed_rev_tokens=word_tokenize(stemmed_rev)
    vocab=[]
    for word in stemmed_rev_tokens:
        if word not in vocab:
            vocab.append(word)
    print(vocab)
    index_word={}
    i = 0   
    for word in vocab:
        index_word[word]=i
        i+=1
    count_dict = defaultdict(int)
    vec = [0 for _ in range(len(vocab))]
    for item in stemmed_rev_tokens:
        count_dict[item] += 1
    for key,item in count_dict.items():
        vec[index_word[key]] = item
    print(vec)
    df2.loc[2].review=vec
    for index,row in df2.iterrows():
        print(row['review'])
    print(df2)
