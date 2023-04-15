from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.visualization import print_metrics

def logisticreg_hyperparameter_search (x_train_bow,y_train):
    #Logistic Regression
    #Hyper-Parameter Optimization
    std_slc = StandardScaler(with_mean=False)

    logistic_reg = LogisticRegression()
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('logistic_reg', logistic_reg)])
    
    C = [0.001, 0.01, 0.1] #inverse of regularization strength -> reduce overfiting by reducing the variance
    solver = ['lbfgs', 'liblinear', 'newton-cg'] #optimization algorithm 
    penalty = ['l1', 'l2'] #type of regularization. 
    parameters = dict(logistic_reg__C=C,
                      logistic_reg__penalty=penalty,
                      logistic_reg__solver=solver)

    clf_gscv= GridSearchCV(pipe, parameters)
    clf_gscv.fit(x_train_bow, y_train)
    print('Best Penalty:', clf_gscv.best_estimator_.get_params()['logistic_reg__penalty'])
    print('Best Solver:', clf_gscv.best_estimator_.get_params()['logistic_reg__solver'])
    print('Best C:', clf_gscv.best_estimator_.get_params()['logistic_reg__C'])
    print(); print(clf_gscv.best_estimator_.get_params()['logistic_reg'])
    
def logisticreg (x_train_bow,y_train, x_test_bow, y_test ):
    lgr=LogisticRegression(C=0.1, penalty='l2', solver='sag')
    lgr.fit(x_train_bow,y_train)
    y_pred_lgr = lgr.predict(x_test_bow)
    print("=============LOGISTIC REGRESSION============")
    metrics=print_metrics(y_test,y_pred_lgr)
    print("============================================")
    return metrics