
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils.visualization import print_metrics
from joblib import dump, load

def randomforest_hyperparamter_search(x_train_bow, y_train):
    rfc = RandomForestClassifier(random_state=4)

    # Definir parámetros a ajustar
    param_grid = {
        'n_estimators': [300, 400, 450, 500, 550, 600],
        'max_depth': [50, 60, 70, 80, 90, 100],
    }

    # Realizar GridSearchCV
    grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1,scoring='accuracy')
    grid_search.fit(x_train_bow, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

def randomforest (x_train_bow,y_train, x_test_bow, y_test ):
    #Random Forest
    rfc=RandomForestClassifier(max_depth=60, random_state=42,min_samples_leaf=1, min_samples_split=2, n_estimators=400)
    rfc.fit(x_train_bow,y_train)
    y_pred_rfc = rfc.predict(x_test_bow)
    print("=============RANDOM FOREST============")
    print_metrics(y_test,y_pred_rfc)
    print("============================================")
    dump(rfc,"rfcmodel.joblib")
       