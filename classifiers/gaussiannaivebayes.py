from sklearn.naive_bayes import GaussianNB
from utils.visualization import print_metrics

def gaussiannaivebayes (x_train_bow,y_train, y_test, x_test_bow ):
    #Gaussian Naive Bayes
    gnbc=GaussianNB()
    gnbc.fit(x_train_bow.toarray(),y_train)
    y_pred_gnbc = gnbc.predict(x_test_bow)
    print("=============GAUSSIAN NAIVE BAYES============")
    print_metrics(y_test,y_pred_gnbc)
    print("============================================")  