from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils.visualization import print_metrics
from joblib import dump, load

def knearestneighbors_hyperparameter_search (x_train_bow,y_train):
 #KNearestNeighbors X -> large execution time
    knn=KNeighborsClassifier(n_jobs=-1)
    param_grid = {
      'n_neighbors': [5, 7, 10, 15],
      'weights': ['uniform', 'distance']
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5,scoring='accuracy')
    grid_search.fit(x_train_bow, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    
def knearestneighbors(x_train_bow,y_train, x_test_bow, y_test):
   #knn=KNeighborsClassifier(n_neighbors=10, weights='uniform')
   knn=load("models/knnmodel.joblib")
   knn.fit(x_train_bow, y_train)
   y_pred_knn = knn.predict(x_test_bow)
   print("=============K-NEAREST NEIGHBORS============")
   metrics=print_metrics(y_test,y_pred_knn)
   print("============================================")
   #dump(knn,"knnmodel.joblib")
    
  