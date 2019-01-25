#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.3)
# dtc.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score
# y_pred = dtc.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy is %.3f" % acc)

pois = filter(lambda x: x==1, y_test)
print("We have %d POIs in the test set" % len(pois))
print("We have total of %d people in the test set" % len(y_test))

true_not_poi = len(y_test) - len(pois)
accuracy_for_all_zeros = true_not_poi / float(len(y_test))
print("If an identifier predicted only zeroes we would have %.2f accuracy" % accuracy_for_all_zeros)