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
import pandas as pd
from utils.vectorize import default_tfidf_vector
from classifiers.logisticregression import logisticreg_hyperparameter_search as lgr_hs
from classifiers.logisticregression import logisticreg as lgr

def training_model():
    x_train_vec,y_train, x_test_bow, y_test = default_tfidf_vector()
    lgr_hs(x_train_vec,y_train)
    #lgr(x_train_vec,y_train,x_test_bow,y_test)

training_model()

