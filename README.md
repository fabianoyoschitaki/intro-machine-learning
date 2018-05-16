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

- http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
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

--------------
Pros and Cons of R vs Python Sci-kit learn
- https://www.kaggle.com/getting-started/5243

--------------
Should you teach Python or R for data science?
- http://www.dataschool.io/python-or-r-for-data-science/

--------------
Unofficial Windows Binaries for Python Extension Packages - Wheels
- http://www.lfd.uci.edu/~gohlke/pythonlibs/

--------------
Top 6 errors novice machine learning engineers make
- https://medium.com/towards-data-science/top-6-errors-novice-machine-learning-engineers-make-e82273d394db

--------------
Becoming a Machine Learning Engineer | Step 2: Pick a Process
- https://medium.com/towards-data-science/becoming-a-machine-learning-engineer-step-2-pick-a-process-942eef6ba8dd

--------------
Becoming a Machine Learning Engineer | Step 3: Pick Your Tool
- https://medium.com/towards-data-science/becoming-a-machine-learning-engineer-step-3-pick-your-tool-da1903a2958f

--------------
Parametric and Nonparametric Machine Learning Algorithms
- https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/

--------------
ML Mind Map
- https://s3.amazonaws.com/MLMastery/MachineLearningAlgorithms.png?__s=tba8qgdaiyahops6sy3p

--------------
My Solution to the Galaxy Zoo Challenge
- http://benanne.github.io/2014/04/05/galaxy-zoo.html

--------------
Object Recognition with Convolutional Neural Networks in the Keras Deep Learning Library
- https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

--------------
TODO TensorFlow Demo: MNIST for ML Beginners
- https://github.com/IanLewis/tensorflow-examples/blob/master/notebooks/TensorFlow%20MNIST%20tutorial.ipynb

--------------
TODO Your First Machine Learning Project in Python Step-By-Step
- https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

--------------
Undertand Bayes Theorem (Posterior, Likelihood, Prior and Evidence)
- http://www.lichun.cc/blog/2013/07/understand-bayes-theorem-prior-likelihood-posterior-evidence/

--------------
Awesome Machine Learning
- https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md

--------------
How do I learn Machine Learning?
- https://www.quora.com/How-do-I-learn-machine-learning-1

--------------
Embrace Randomness in Machine Learning
- https://machinelearningmastery.com/randomness-in-machine-learning/

--------------
Redes Neurais Artificiais
- http://conteudo.icmc.usp.br/pessoas/andre/research/neural/

--------------
https://pt.stackoverflow.com/questions/192098/como-funciona-uma-rede-neural-artificial
https://pt.stackoverflow.com/questions/61187/como-implementar-a-camada-oculta-em-uma-rede-neural-de-reconhecimento-de-caracte
https://pt.stackoverflow.com/questions/40135/explicar-o-algoritmo-svr/40149#40149
https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks

Análise de Componentes Principais
- http://prorum.com/index.php/1557/estatistica-conhecido-componentes-principais-principal

http://iamtrask.github.io/2015/07/12/basic-python-network/


--------------
Deep Learning Book
- http://www.deeplearningbook.org/

--------------
Keras Cheat Sheet: Neural Networks in Python
- https://www.datacamp.com/community/blog/keras-cheat-sheet
- https://www.datacamp.com/community/tutorials/deep-learning-python
- https://www.datacamp.com/community/tutorials/keras-r-deep-learning

--------------
How to Setup a Python Environment for Machine Learning and Deep Learning with Anaconda
- https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/

--------------
Building a Student Intervention System
- http://www.ritchieng.com/machine-learning-project-student-intervention/

--------------
Machine Learning Algorithms: Which One to Choose for Your Problem
- https://blog.statsbot.co/machine-learning-algorithms-183cc73197c

--------------
If you have already closed the Anaconda navigator, open cmd and type jupyter-notebook list.

Then you can kill the port using following commands:
netstat -o -n -a | findstr :3000
   TCP    0.0.0.0:3000      0.0.0.0:0              LISTENING       3116
taskkill /F /PID 3116

- https://stackoverflow.com/questions/46319617/how-to-stop-jupyter-server-using-anaconda

--------------
Comparison of 14 different families of classification algorithms on 115 binary datasets
- https://arxiv.org/pdf/1606.00930.pdf

