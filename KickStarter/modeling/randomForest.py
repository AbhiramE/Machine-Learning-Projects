import math
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import ensemble
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_training_set_and_labels():
    df = pd.read_csv('../final_train.csv')
    df = df.drop(df.columns[[0]], 1)
    X_tr = df.ix[:, df.columns != 'final_status']
    y_tr = df.ix[:, df.columns == 'final_status']
    return X_tr.as_matrix(), y_tr.as_matrix()


def get_test_set_and_project_ids():
    df = pd.read_csv('../final_test.csv')
    df = df.drop(df.columns[[0]], 1)
    X_test = df.ix[:, df.columns != 'project_id']
    project_ids = df.ix[:, df.columns == 'project_id']
    return X_test.as_matrix(), project_ids.as_matrix()


def select_features(trainX, trainY, testX, testY=None):
    selector = SelectKBest(k=11)
    selector.fit(trainX, trainY)
    trainX = selector.fit_transform(trainX, trainY)
    testX = selector.transform(testX)

    if testY is None:
        return trainX, trainY.as_matrix().reshape(len(trainY), 1), testX
    else:
        return trainX, trainY.as_matrix().reshape(len(trainY), 1), testX, testY.as_matrix().reshape(len(testY), 1)


def hyperparameter_selection(trainX, trainY, param_grid, method):
    print "In hyperparameter selection now"
    trainY = trainY.reshape(len(trainY), )
    clf = GridSearchCV(method, param_grid)
    clf.fit(trainX, trainY)
    return clf


def classify(trainX, trainY, testX, project_ids, best):
    # n_estimators = best.best_params_['n_estimators']
    clf = ensemble.RandomForestClassifier(n_estimators=2000)
    clf.fit(trainX, trainY)
    predictions = clf.predict(testX)
    f = open('submission_random_forest.csv', 'w')
    for i in range(0, len(testX)):
        f.write("\"" + str(project_ids[i][0]) + "\"," + str(predictions[i]) + "\n")
    f.close()


def validate(trainX, trainY, testX, testY, best):
    n_estimators = best.best_params_['n_estimators']
    print n_estimators
    clf = ensemble.RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(trainX, trainY)
    predictY = clf.predict(testX)
    print accuracy_score(testY, predictY)


def pipeline():
    # X_tr, y_tr = get_training_set_and_labels()
    # X_train, X_validate, y_train, y_validate = train_test_split(X_tr, y_tr, test_size=0.2)
    # param_grid = {"n_estimators": [1000, 2000, 3000]}
    # best = hyperparameter_selection(X_train, y_train, param_grid=param_grid,
    #                                method=ensemble.RandomForestClassifier(random_state=44))
    best = []
    X_tr, y_tr = get_training_set_and_labels()
    X_test, project_ids = get_test_set_and_project_ids()
    # validate(X_train, y_train, X_validate, y_validate, best)
    classify(X_tr, y_tr, X_test, project_ids.reshape(len(project_ids), 1), best)


pipeline()
