from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.visualization import print_metrics
from joblib import dump, load

def logisticreg_hyperparameter_search (x_train_bow,y_train):
    #Logistic Regression
    #Hyper-Parameter Optimization
    std_slc = StandardScaler(with_mean=False)

    logistic_reg = LogisticRegression()
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('logistic_reg', logistic_reg)])
    
    C = [0.001, 0.01, 0.1, 1, 10, 100] #inverse of regularization strength -> reduce overfiting by reducing the variance
    solver = ['lbfgs', 'liblinear', 'newton-cg', 'saga'] #optimization algorithm 
    penalty = ['none','l1', 'l2'] #type of regularization. 
    parameters = dict(logistic_reg__C=C,
                      logistic_reg__penalty=penalty,
                      logistic_reg__solver=solver)

    grid_search= GridSearchCV(pipe, parameters,cv=5,scoring='accuracy')
    grid_search.fit(x_train_bow, y_train)
    print('Best Penalty:', grid_search.best_estimator_.get_params()['logistic_reg__penalty'])
    print('Best Solver:', grid_search.best_estimator_.get_params()['logistic_reg__solver'])
    print('Best C:', grid_search.best_estimator_.get_params()['logistic_reg__C'])
    print(); print(grid_search.best_estimator_.get_params()['logistic_reg'])
    
def logisticreg (x_train_bow,y_train, x_test_bow, y_test ):
    #lgr=LogisticRegression(C=1, penalty='l1', solver='liblinear')
    lgr=load("models/lgrmodel.joblib")
    lgr.fit(x_train_bow,y_train)
    y_pred_lgr = lgr.predict(x_test_bow)
    print("=============LOGISTIC REGRESSION============")
    metrics=print_metrics(y_test,y_pred_lgr)
    print("============================================")
    #dump(lgr,"lgrmodel.joblib")
    return metrics