from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils.visualization import print_metrics

def knearestneighbors (x_train_bow,y_train):
 #KNearestNeighbors X -> large execution time
    knn=KNeighborsClassifier(n_jobs=-1)
    k_range=list(range(1,50))
    options=['uniform', 'distance']
    param_grid = dict(n_neighbors=k_range, weights=options)
    rand_knn = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10)
    rand_knn.fit(x_train_bow, y_train)
    print(rand_knn.best_score_)
    print(rand_knn.best_params_)