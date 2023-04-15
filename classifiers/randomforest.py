
from sklearn.ensemble import RandomForestClassifier
from utils.visualization import print_metrics


def randomforest (x_train_bow,y_train, x_test_bow, y_test ):
    #Random Forest
    rfc=RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(x_train_bow,y_train)
    y_pred_rfc = rfc.predict(x_test_bow)
    print("=============RANDOM FOREST============")
    print_metrics(y_test,y_pred_rfc)
    print("============================================")   