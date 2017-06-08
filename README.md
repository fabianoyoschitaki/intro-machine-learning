ud120-projects - https://br.udacity.com/course/intro-to-machine-learning--ud120/
==============
Python code for Udacity Introduction to Machine Learning course. 
Some code files were written by myself in order to achieve different results from the given tests. 

Get started (check course link)

- Download enron mail dataset at: https://www.cs.cmu.edu/~./enron/
- Run: python tools/startup.py
- fabiano_tutoriais folder has some random files which helped me to understand most of the initial concepts

--------------

Links:

Class imbalance problem: 
- http://www.chioka.in/class-imbalance-problem/
- http://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation
- https://gallery.cortanaintelligence.com/Experiment/Online-Fraud-Detection-working-with-unbalanced-class-data-1
- https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/
- http://storm.cis.fordham.edu/~gweiss/papers/dmin07-weiss.pdf
- http://www.jair.org/media/953/live-953-2037-jair.pdf
- http://abricom.org.br/wp-content/uploads/2016/03/bricsccicbic2013_submission_20.pdf
	
--------------
Scikit contrib on imbalaced data
- https://github.com/scikit-learn-contrib/imbalanced-learn

--------------
- http://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html

That said, here is a rough outline of useful approaches. These are listed approximately in order of effort:

- Do nothing. Sometimes you get lucky and nothing needs to be done. You can train on the so-called natural (or stratified) distribution and sometimes it works without need for modification.
- Balance the training set in some way:
  - Oversample the minority class.
  - Undersample the majority class.
  - Synthesize new minority classes.
- Throw away minority examples and switch to an anomaly detection framework.
- At the algorithm level, or after it:
  - Adjust the class weight (misclassification costs).
  - Adjust the decision threshold.
  - Modify an existing algorithm to be more sensitive to rare classes.
- Construct an entirely new algorithm to perform well on imbalanced data.

--------------
Ensemble method Machine Learning
- https://www.toptal.com/machine-learning/ensemble-methods-machine-learning

--------------
Basic Concepts in Machine Learning
- http://machinelearningmastery.com/basic-concepts-in-machine-learning/

--------------
Python begginer - Code Academy
- https://www.codecademy.com/courses/python-beginner-pt-BR

--------------
Introducing: Machine Learning in R
- https://www.datacamp.com/community/tutorials/machine-learning-in-r#gs.qyPOXoU

--------------
Your First Machine Learning Project in R Step-By-Step (tutorial and template for future projects)
- http://machinelearningmastery.com/machine-learning-in-r-step-by-step/

--------------
Python vs R for machine learning
- https://datascience.stackexchange.com/questions/326/python-vs-r-for-machine-learning