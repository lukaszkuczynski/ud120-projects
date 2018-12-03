#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print("We have %d people in the dataset" % len(enron_data.keys()))

print("For the first person we have %d features" % len(enron_data.items()[0][1]))

poi = filter(lambda a: a['poi'], enron_data.values())
print("In whole dataset we have %d POIs" % len(poi))

with open("../final_project/poi_names.txt") as f:
    for i,l in enumerate(f):
        pass
    i = i + 1
total_name_lines_count = i - 2
print("In txt file we have %d POI" % total_name_lines_count)

print("Mr James has total stock of %d" % enron_data["PRENTICE JAMES"]['total_stock_value'])
print("Mr Wesley sent %d emails to POIs" % enron_data["COLWELL WESLEY"]['from_this_person_to_poi'])
