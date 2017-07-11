import math
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


def select_features(trainX, trainY, testX, testY, k):
    clf = KNeighborsClassifier()
    if k == 14:
        k = 'all'
    selector = SelectKBest(k=k)
    selector.fit(trainX, trainY)
    trainX = selector.fit_transform(trainX, trainY)
    testX = selector.transform(testX)
    print "Selected features", selector.get_support()

    clf.fit(X=trainX, y=trainY)
    y_predict = clf.predict(testX)
    score = accuracy_score(testY, y_predict)
    print "Feature Selection RMSE ", k, score
    return score


def get_training_set_and_labels():
    df = pd.read_csv('final_train.csv')
    df = df.drop(df.columns[[0]], 1)
    X_tr = df.ix[:, df.columns != 'final_status']
    y_tr = df.ix[:, df.columns == 'final_status']
    return X_tr, y_tr


X_tr, y_tr = get_training_set_and_labels()
X_train, X_validate, y_train, y_validate = train_test_split(X_tr, y_tr, test_size=0.2)

accuracies = []

'''
for k in range(6, 15):
    accuracies.append(select_features(X_train, y_train, X_validate, y_validate, k))

print max(accuracies), " k = " + str(accuracies.index(max(accuracies)))
'''


print X_train.columns
select_features(X_train, y_train, X_validate, y_validate, k=11)
