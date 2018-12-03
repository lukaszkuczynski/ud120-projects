#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


def getSVM(features, labels):
    from sklearn.svm import OneClassSVM, LinearSVC, SVC
    clff = SVC(kernel='rbf', C=10000, gamma=2000)
    clff.fit(features, labels)
    return clff

def getTree(features_train, labels_train, split=2):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(min_samples_split=split)
    dtc.fit(features_train, labels_train)
    return dtc


def getKMeans(features_train, labels_train, no_of_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(no_of_clusters, init='random', random_state=0)
    kmeans.fit(features_train, labels_train)
    return kmeans

def getRandomForest(features_train, labels_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=0)
    clf.fit(features_train, labels_train)
    return clf

def getAdaBoost(features_train, labels_train, n_estimators=50, learning_rate=1.0):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(features_train, labels_train)
    return clf

#clf = getSVM(features_train, labels_train)
#clf2 = getTree(features_train, labels_train, 2)
#clf50 = getTree(features_train, labels_train, 50)

from sklearn.metrics import accuracy_score
import numpy as np

#n_clusters = len(np.unique(labels_train))
#kmeans = getKMeans(features_train, labels_train, no_of_clusters=n_clusters)
#y_pred_kmeans = kmeans.predict(features_test)
#acc_kmeans = accuracy_score(labels_test, y_pred_kmeans)

#forest = getRandomForest(features_train, labels_train)
#y_pred_forest = forest.predict(features_test)
#acc_forest = accuracy_score(labels_test, y_pred_forest)

def draw_matrix(matrix):
    elems = sorted(matrix.items())
    x,y = zip(*elems)
    plt.plot(x, y)
    print("max for plot is %f" % max(y))
    plt.show()    

import numpy as np
accuracies = {}

def many_ada_versions():
    for param_estimators in range(10,30,1):
        for param_rate in np.linspace(1,3,5):
            param_rate = 2
            ada = getAdaBoost(features_train, labels_train, n_estimators=param_estimators, learning_rate=param_rate)
            y_pred_ada = ada.predict(features_test)
            acc_ada = accuracy_score(labels_test, y_pred_ada)
            yield {
                    'learning_rate': param_rate,
                    'n_estimators': param_estimators,
                    'accuracy': acc_ada
            }

#import pandas as pd
#df = pd.DataFrame(many_ada_versions())
#df_pivot = df.pivot(index='n_estimators', columns='learning_rate', values='accuracy')
#df_pivot.plot()
#print(df_pivot.max())

#the best score: ADA : n=20, rate=2
ada = getAdaBoost(features_train, labels_train, n_estimators=20, learning_rate=2)
y_pred_ada = ada.predict(features_test)
acc_ada = accuracy_score(labels_test, y_pred_ada)
acc = {
#        "acc_kmeans": round(acc_kmeans,3),   
#        "acc_forest": round(acc_forest,3),
        "acc_ada": round(acc_ada,3),
}
print(acc)

try:
    prettyPicture(ada, features_test, labels_test)
    plt.show()
except NameError:
    pass
