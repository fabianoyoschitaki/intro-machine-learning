#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
	"""
		Clean away the 10% of points that have the largest
		residual errors (difference between the prediction
		and the actual net worth).

		Return a list of tuples named cleaned_data where 
		each tuple is of the form (age, net_worth, error).
	"""
	
	cleaned_data = []

	### your code goes here
	
	error = list((net_worths - predictions)**2) #squared error	
	#print "error variable is:" + str(type(error))
	#cont = 0
	#for err in error:
	#	print str(err) + " = " + str(net_worths[cont]) + "-" + str(predictions[cont])
	#	cont += 1
	
	#concatenate arrays
	cleaned_data = zip(ages, net_worths, error)
	#print "cleaned_data: " + str(type(cleaned_data)) + str(cleaned_data[0])
	#for item in cleaned_data:
	#	print str(item)
	
	#sort data by error ascending
	cleaned_data = sorted(cleaned_data, key = lambda tup: tup[2])
	#print "cleaned_data sorted by tup[2]: " + str(type(cleaned_data)) + str(cleaned_data[0])
	#for item in cleaned_data:
	#	print str(item)
		
	#keep first 80 records
	cleaned_data = cleaned_data[:80]
	return cleaned_data

