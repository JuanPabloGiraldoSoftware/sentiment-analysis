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
from utils.vectorize import default_tfidf_vector
from utils.preprocessing import build_datasets
from classifiers.logisticregression import logisticreg_hyperparameter_search as lgr_hs
from classifiers.logisticregression import logisticreg as lgr
from classifiers.knearestneighbors import knearestneighbors as knn
from classifiers.knearestneighbors import knearestneighbors_hyperparameter_search as knn_hs
from classifiers.randomforest import randomforest as rfc
from classifiers.randomforest import randomforest_hyperparamter_search as rfc_hs
from classifiers.supportvm import supportvm as svm
from classifiers.supportvm import supportvm_hyperparameter_search as svm_hs
from DeepL.rnn_template import rnn_network


#build_datasets()

def train_models():
    x_train_vec,y_train, x_test_vec, y_test = default_tfidf_vector(10000)
    #Logistec Regression Classifier
    #lgr_hs(x_train_vec,y_train)
    lgr(x_train_vec,y_train,x_test_vec,y_test)
    
    #K-Nearest Neighbors Classifier
    #knn_hs(x_train_vec,y_train)
    knn(x_train_vec, y_train, x_test_vec, y_test)
    
    #Random Forest Classifier
    #rfc_hs(x_train_vec,y_train)
    rfc(x_train_vec, y_train, x_test_vec, y_test)
    
    #Support Vector Machine
    #svm_hs(x_train_vec,y_train)
    svm(x_train_vec, y_train, x_test_vec, y_test)

    #rnn_network(x_train_vec, y_train, x_test_vec, y_test)


train_models()
