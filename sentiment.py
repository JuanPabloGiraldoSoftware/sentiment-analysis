#Author: Juan Pablo Giraldo M.#

###################################################
#          Raw datasets sources                   #
###################################################
#movies1.csv Link: https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
#movies2.csv Link: https://www.kaggle.com/code/yasserh/imdb-movie-rating-sentiment-analysis/data

#Drive with all csv datasets: https://drive.google.com/drive/folders/1suuC9UCK2SY4qtHnn4L7Jubbh-b55ZL9?usp=sharing

###################################################
#          libraries and external functions       #
###################################################
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re #To remove non-alphabetic characters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error

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

def print_confm(confm):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confm, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(confm.shape[0]):
        for j in range(confm.shape[1]):
            ax.text(x=j, y=i,s=confm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def print_metrics(y_test,y_pred):
    pscore = precision_score(y_test, y_pred)*100
    rscore = recall_score(y_test, y_pred)*100
    ascore = accuracy_score(y_test, y_pred)*100
    fscore = f1_score(y_test, y_pred)*100
    print('Precision: %.3f' % pscore)
    print('Recall: %.3f' % rscore)
    print('Accuracy: %.3f' % ascore)
    print('F1 Score: %.3f' % fscore)

def print_linear_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Absolute Error: %.3f' % mae)
    print('Mean Squared Error: %.3f' % mse)
    print('Root Mean Squared Deviation: %.3f' % rmse)

def training_model():
    df = pd.read_csv("clean_rev_movies.csv")
    print(df.shape)
    df = df.loc[:40000]
    X=df['review']
    Y=df['sentiment']
    print(X)
    print(Y)
    print("###################################################")
    vectorizer = TfidfVectorizer() #Better performance than CountVectorizer in exchange of computational cost
    x_train, x_test, y_train, y_test = train_test_split(X,Y,stratify=Y, test_size=0.30)
    print_shape(x_train,x_test)
    x_train_bow=vectorizer.fit_transform(x_train)
    x_test_bow=vectorizer.transform(x_test)
    print_shape(x_train_bow,x_test_bow)

    #KNearestNeighbors X -> large execution time
    #knn=KNeighborsClassifier()
    #k_range=list(range(1,50))
    #options=['uniform', 'distance']
    #param_grid = dict(n_neighbors=k_range, weights=options)
    #rand_knn = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=0)
    #rand_knn.fit(x_train_bow, y_train)
    #print(rand_knn.best_score_)
    #print(rand_knn.best_params_)
    #confm_knn = confusion_matrix(y_test, y_pred_knn)
    #print_confm(confm_knn)
    #print("=============K NEAREST NEIGHBORS============")
    #print_metrics(y_test,y_pred_knn)
    #print("============================================")

    #Logistic Regression
    lgr=LogisticRegression(random_state=0)
    lgr.fit(x_train_bow,y_train)
    y_pred_lgr = lgr.predict(x_test_bow)
    print("=============LOGISTIC REGRESSION============")
    print_metrics(y_test,y_pred_lgr)
    print("============================================")

    #Linear Regression
    lnr=LinearRegression()
    lnr.fit(x_train_bow,y_train)
    y_pred_lnr = lnr.predict(x_test_bow)
    print("=============LINEAR REGRESSION============")
    print_linear_metrics(y_test,y_pred_lnr)
    print("============================================")   

    #Random Forest
    rfc=RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(x_train_bow,y_train)
    y_pred_rfc = rfc.predict(x_test_bow)
    print("=============RANDOM FOREST============")
    print_metrics(y_test,y_pred_rfc)
    print("============================================")   

    #Decision Tree
    dtc=DecisionTreeClassifier()
    dtc.fit(x_train_bow,y_train)
    y_pred_dtc = dtc.predict(x_test_bow)
    print("=============DECISION TREE============")
    print_metrics(y_test,y_pred_dtc)
    print("============================================")   

    #Support Vector Macine  X -> large execution time
    #svc=SVC(C = 100, kernel = 'linear', random_state=123)
    #svc.fit(x_train_bow,y_train)
    #y_pred_svc = svc.predict(x_test_bow)
    #print("=============SUPPORT VECTOR MACHINE============")
    #print_metrics(y_test,y_pred_svc)
    #print("============================================")   

    #Gaussian Naive Bayes
    gnbc=GaussianNB()
    gnbc.fit(x_train_bow.toarray(),y_train)
    y_pred_gnbc = gnbc.predict(x_test_bow)
    print("=============GAUSSIAN NAIVE BAYES============")
    print_metrics(y_test,y_pred_gnbc)
    print("============================================")   

def testing_k_neighbors(x_train_bow,y_train,x_test_bow,y_test):
    accuracy_hist = []
    for i in range (1,21):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train_bow, y_train)
        yi_pred_knn = knn.predict(x_test_bow)
        acc_i = accuracy_score(y_test, yi_pred_knn)
        accuracy_hist.append(acc_i)
        print(f"K: {i}, accuracy: {acc_i}")
    print(accuracy_hist)

def plot_knn_results():
    k_axis = [i for i in range (1,21)]
    acc =[0.7384384634613782, 0.7435213732188984, 0.7574368802599784, 0.7678526789434214, 
    0.7681859845012916, 0.7745187901008249, 0.7729355887009416, 0.7774352137321889, 0.7742688109324223, 
    0.7810182484792934, 0.7776851929005916, 0.7854345471210732, 0.783101408215982, 0.7866844429630864, 
    0.784934588784268, 0.78860094992084, 0.7873510540788268, 0.7893508874260479, 0.7856011999000083, 0.7916006999416715]
    #x_ticks=np.arange(1,21,1)
    #y_ticks= np.arange (0.7, 0.8, 0.01)
    #plt.xticks(x_ticks)
    #plt.yticks(y_ticks)
    plt.plot(k_axis,acc)
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.show()
    
#plot_knn_results()
training_model()


