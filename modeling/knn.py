import math
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_training_set_and_labels():
    df = pd.read_csv('../final_train.csv')
    df = df.drop(df.columns[[0]], 1)
    X_tr = df.ix[:, df.columns != 'final_status']
    y_tr = df.ix[:, df.columns == 'final_status']
    return X_tr, y_tr


def get_test_set_and_project_ids():
    df = pd.read_csv('../final_test.csv')
    df = df.drop(df.columns[[0]], 1)
    X_test = df.ix[:, df.columns != 'project_id']
    project_ids = df.ix[:, df.columns == 'project_id']
    return X_test, project_ids


def select_features(trainX, trainY, testX, testY=None):
    selector = SelectKBest(k=11)
    selector.fit(trainX, trainY)
    trainX = selector.fit_transform(trainX, trainY)
    testX = selector.transform(testX)

    if testY is None:
        return trainX, trainY.as_matrix().reshape(len(trainY), 1), testX
    else:
        return trainX, trainY.as_matrix().reshape(len(trainY), 1), testX, testY.as_matrix().reshape(len(testY), 1)


def hyperparameter_selection(trainX, trainY, testX, testY):
    neighborList = list(range(200, 400))
    neighbors = filter(lambda x: x % 2 != 0, neighborList)
    cv_scores = []

    for k in neighbors:
        print k
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, trainX, trainY.ravel(), cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    print cv_scores

    return 200 + cv_scores.index(max(cv_scores)) + 1


def classify(trainX, trainY, testX, project_ids):
    clf = KNeighborsClassifier(n_neighbors=290)
    clf.fit(trainX, trainY)

    print trainX.shape
    test_matrix = []
    for i in range(0, len(testX)):
        prediction = clf.predict(testX[i])
        row = np.array([str(project_ids[i][0]), int(prediction[0])])

        if len(test_matrix) == 0:
            test_matrix = row
        else:
            test_matrix = np.vstack([test_matrix, row])

    np.savetxt('submission.csv', test_matrix, delimiter=',')


def pipeline():
    X_tr, y_tr = get_training_set_and_labels()
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, y_tr, test_size=0.2)
    X_train, y_train, X_validate, y_validate = select_features(X_train, y_train, X_validate, y_validate)
    # optimum_neighbours = hyperparameter_selection(X_train, y_train, X_validate, y_validate)

    X_tr, y_tr = get_training_set_and_labels()
    X_test, project_ids = get_test_set_and_project_ids()
    X_tr, y_tr, X_test = select_features(X_tr, y_tr, X_test)
    classify(X_tr, y_tr, X_test, project_ids.as_matrix().reshape(len(project_ids), 1))


pipeline()
