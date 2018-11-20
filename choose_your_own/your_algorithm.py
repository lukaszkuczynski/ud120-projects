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
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


def getSVM(features, labels):
    from sklearn.svm import OneClassSVM, LinearSVC, SVC
    clff = SVC(kernel='rbf', C=10000, gamma=2000)
    clff.fit(features, labels)
    return clff

def getTree(features_train, labels_train):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(features_train, labels_train)
    return dtc

def submitAccuracies(clf, features_test, labels_test):
    from sklearn.metrics import accuracy_score
    y_pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, y_pred)
    return {"acc":round(acc,3)}

#clf = getSVM(features_train, labels_train)
clf = getTree(features_train, labels_train)
# clf = None
acc = submitAccuracies(clf, features_test, labels_test)
print(acc)

#try:
#    prettyPicture(clf, features_test, labels_test)
#    plt.show()
#except NameError:
#    pass
