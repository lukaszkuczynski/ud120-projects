#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

not_nan_salaries = filter(lambda person:person[1]['salary'] != 'NaN', data_dict.items())
float_salaries = map(lambda person:(person[0], float(person[1]['salary'])), not_nan_salaries)
sorted_salaries = sorted(float_salaries, key=lambda x:x[1], reverse=True)
print(sorted_salaries[:5])

not_nan_bonuses = filter(lambda person:person[1]['bonus'] != 'NaN', data_dict.items())
float_bonuses = map(lambda person:(person[0], float(person[1]['bonus']), float(person[1]['salary'])), not_nan_bonuses)
sorted_bonuses = sorted(float_bonuses, key=lambda x:x[1], reverse=True)
print(sorted_bonuses[:5])