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

def getTree(features_train, labels_train, split=2):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(min_samples_split=split)
    dtc.fit(features_train, labels_train)
    return dtc




#clf = getSVM(features_train, labels_train)
clf2 = getTree(features_train, labels_train, 2)
clf50 = getTree(features_train, labels_train, 50)

from sklearn.metrics import accuracy_score
y_pred = clf2.predict(features_test)
acc_min_samples_split_2 = accuracy_score(labels_test, y_pred)
y_pred = clf50.predict(features_test)
acc_min_samples_split_50 = accuracy_score(labels_test, y_pred)
acc = {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
print(acc)

#try:
#    prettyPicture(clf2, features_test, labels_test)
#    prettyPicture(clf50, features_test, labels_test)
#    plt.show()
#except NameError:
#    pass
