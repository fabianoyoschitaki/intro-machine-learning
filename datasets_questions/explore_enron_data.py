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

#https://docs.python.org/2/tutorial/datastructures.html#dictionaries
print type(enron_data)

#How many data points (people) are in the dataset?
print len(enron_data)

#Print all person (key)
print enron_data.keys()

#For each person, how many features are available?

#for key in enron_data.keys():
	#print type(enron_data.get(key)) dict
	#print len(enron_data.get(key)) #21
	#print enron_data.get(key) # all data


#The "poi" feature records whether the person is a person of interest, according to our definition. How many POIs are there in the E+F dataset?
#In other words, count the number of entries in the dictionary where data[person_name]["poi"]==1
contPoi = 0;
for key in enron_data.keys():
	if (enron_data.get(key)["poi"] == 1):
		contPoi = contPoi + 1
print "contPoi:", contPoi

#Like any dict of dicts, individual people/features can be accessed like so:
#enron_data["LASTNAME FIRSTNAME"]["feature_name"] or, sometimes enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]
#What is the total value of the stock belonging to James Prentice?
#print enron_data["PRENTICE JAMES"]
print enron_data["PRENTICE JAMES"]["total_stock_value"]

#How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

#What's the value of stock options exercised by Jeffrey K Skilling?
print enron_data["SKILLING JEFFREY K"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

#Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of "total_payments" feature)?
#How much money did that person get?
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]

#How many folks in this dataset have a quantified salary? What about a known email address?
contQuantifiedSalary = 0;
contEmailAddress = 0;
for key in enron_data.keys():
	if (enron_data.get(key)["salary"] != 'NaN'):
		contQuantifiedSalary = contQuantifiedSalary + 1
	if (enron_data.get(key)["email_address"] != 'NaN'):
		contEmailAddress = contEmailAddress + 1
print "contQuantifiedSalary:", contQuantifiedSalary
print "contEmailAddress:", contEmailAddress

#### Dict to Array Conversion ###
#A python dictionary can't be read directly into an sklearn classification or regression algorithm; instead, it needs a numpy array or a list of lists (each element of the list (itself a list) is a data point, and the elements of the smaller list are the features of that point).
#We've written some helper functions (featureFormat() and targetFeatureSplit() in tools/feature_format.py) that can take a list of feature names and the data dictionary, and return a numpy array.
#In the case when a feature does not have a value for a particular person, this function will also replace the feature value with 0 (zero).