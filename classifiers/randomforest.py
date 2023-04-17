
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils.visualization import print_metrics

def randomforest_hyperparamter_search(x_train_bow, y_train):
    rfc = RandomForestClassifier(random_state=42)

    # Definir parámetros a ajustar
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Realizar GridSearchCV
    grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(x_train_bow, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

def randomforest (x_train_bow,y_train, x_test_bow, y_test ):
    #Random Forest
    rfc=RandomForestClassifier(max_depth=30, random_state=42,min_samples_leaf=1, min_samples_split=2, n_estimators=200)
    rfc.fit(x_train_bow,y_train)
    y_pred_rfc = rfc.predict(x_test_bow)
    print("=============RANDOM FOREST============")
    print_metrics(y_test,y_pred_rfc)
    print("============================================")   