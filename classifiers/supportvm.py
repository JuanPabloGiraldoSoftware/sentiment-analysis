from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils.visualization import print_metrics

def supportvm_hyperparameter_search(x_train_bow,y_train):
    svm = SVC(kernel='rbf', C=1)

    param_grid = {
        'gamma': [0.0001,0.001,0.01,0.1, 1],
    }

    grid_search = GridSearchCV(svm, param_grid=param_grid, cv=5, n_jobs=-1,scoring='accuracy')
    grid_search.fit(x_train_bow, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

def supportvm (x_train_bow,y_train, x_test_bow, y_test ):
    #Support Vector Macine  X -> large execution time
    svc=SVC(C = 1, kernel = 'rbf', gamma = 0.1, random_state=42)
    svc.fit(x_train_bow,y_train)
    y_pred_svc = svc.predict(x_test_bow)
    print("=============SUPPORT VECTOR MACHINE============")
    print_metrics(y_test,y_pred_svc)
    print("============================================")   