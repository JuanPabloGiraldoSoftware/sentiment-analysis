from sklearn.svm import SVC
from utils.visualization import print_metrics

def supportvm (x_train_bow,y_train, x_test_bow, y_test ):
    #Support Vector Macine  X -> large execution time
    svc=SVC(C = 100, kernel = 'linear', random_state=123)
    svc.fit(x_train_bow,y_train)
    y_pred_svc = svc.predict(x_test_bow)
    print("=============SUPPORT VECTOR MACHINE============")
    print_metrics(y_test,y_pred_svc)
    print("============================================")   