def studentReg(ages_train, net_worths_train, ages_test, net_worths_test):
	### import the sklearn regression module, create, and train your regression
	from sklearn import linear_model

	### name your regression reg
	reg = linear_model.LinearRegression()
		
	### your code goes here!
	reg.fit(ages_train, net_worths_train)   

	print "\nfabiano prediction:", reg.predict([[29]]) 
	print "\nslope:", reg.coef_
	print "\nintercept:", reg.intercept_

	print "\n#### stats on test dataset ####"
	print "\nr-squared score:", reg.score(ages_test, net_worths_test)

	print "\n#### stats on training dataset ####"
	print "\nr-squared score:", reg.score(ages_train, net_worths_train)

	return reg