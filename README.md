# Machine-Learning

Q1.K-Nearest Neighbors

      1. Implement a KNN based classifier to predict digits from images of handwritten
      digits in the dataset.
      2. Featurize the images as vectors that can be used for classification.
      3. Experiment with different values of K(number of neighbors).
      4. Experiment with different distance measures - Euclidean distance, Manhattan dis-
      tance,
      5. Report accuracy score, F1-score, Confusion matrix and any other metrics you feel
      useful.
      6. Implement baselines such as random guessing/majority voting and compare perfor-
      mance. Also, report the performance of scikit-learn’s kNN classifier. Report your
      findings.
      
Q2. K-Nearest Neighbors

      1. Implement a KNN based classifier to classify given set of features in Mushroom
      Database. Missing data must be handled appropriately.(Denoted by ”?”).
      2. Choose an appropriate distance measure for categorical features.
      3. Experiment with different values of K(number of neighbors).
      4. Report accuracy score, F1-score, Confusion matrix and any other metrics you feel
      useful.
      5. Implement baselines such as random guessing/majority voting and compare perfor-
      mance. Also, report the performance of scikit-learn’s kNN classifier. Report your
      findings.
      
Q3. Decision Tree

      1. Implement a decision tree to predict housing prices for the given dataset using the
      available features.
      2. The various attributes of the data are explained in the file data description.txt.
      Note that some attributes are categorical while others are continuos.
      3. Feel Free to use Python libraries such as binarytree or any other library in Python
      to implement the binary tree. However, you cannot use libraries like scikit-learn
      which automatically create the decision tree for you.
      4. Use variance reduction as the criterion for choosing the split in the decision tree.
      Experiment with different approaches to decide when to terminate the tree.
      5. Report metrics such as Mean Squared Error(MSE) and Mean Absolute Error(MAE)
      along with any other metrics that you feel may be useful.
      6. For feature engineering, you may consider normalizing/standardizing the data.SMAI (CSE/ECE 478)
      7. Implement simple baselines such as always predicting the mean/median of the train-
      ing data. Also, compare the performance against scikit-learn’s decision tree. Report
      your findings.
    
Q4. Gussian Mixture Models Clustering
      
      1. You are given 3 data files(dataset1.pkl,dataset2.pkl,dataset3.pkl) and 1 code file
      gmm.py. The code consists of -
      (a) Function to load dataset.
      (b) Function to save dataset.
      (c) Class GMM1D which consists multiple functions.
      2. Load dataset .
      3. Use inbuilt sklearn functions to cluster(GMM clustering) the points and plot them.
      Also report no of iterations taken to converge.
      4. In GMM1D, fill in the blanks with code and cluster the points. Plot for each
      iteration.

Q5. Linear Regression

      1. Given a NASA data set, obtained from a series of aerodynamic and acoustic tests
      of two and three-dimensional airfoil blade sections. Implement a linear regression
      model from scratch using gradient descent to predict scaled sound pressure level.
      The various attributes of the data are explained in the file description.txt.
      2. Using appropriate plot show how number of iterations is affecting the mean squared
      error for above model under below given conditions:
      (a) Using 3 different initial regression coefficients (weights) for fixed value of learn-
      ing parameter (All 3 in single plot).SMAI (CSE/ECE 471)
      Assignment 2 - Page 3 of 4
      Posted: 16/02/2020
      (b) Using 3 different learning parameters for some fixed initial regression coeffi-
      cients. (All 3 in single plot)
     
Q6. Linear Regression

      1. Given a dataset containing historical weather information of certain area, imple-
      ment a linear regression model from scratch using gradient descent to predict the
      apparent temperature. The various attributes of the data are explained in the file
      description.txt. Note that attributes are text, categorical as well as continuous.
      Note: Test data will have 10 columns. Apparent temperature column will be
      missing from in between.
      2. Compare the performance of different error functions ( Mean square error, Mean
      Absolute error, Mean absolute percentage error) and explain the reasons for the
      observed behaviour.
      3. Analyse and report the behaviour of the regression coefficients(for example: sign
      of coefficients, value of coefficients etc.) and support it with appropriate plots as
      necessary.

Q7. Support Vector Machine

      1. Given a dataset which contains a excerpts of text written by some author and the
      corresponding author tag, implement an SVM classifier to predict the author tag
      of the test text excerpts.
      2. For the feature extraction of the text segments, either use Vectorizers provided in
      sklearn or use pre-trained word embedding models. ( Code snippet for usage of
      word embedding models is given here).
      3. Visualize the feature vectors and see if you could find some pattern.
      4. Tweak different parameters of the Linear SVM and report the results.
      5. Experiment different kernels for classification and report the results.
      6. Report accuracy score, F1-score, Confusion matrix and any other metrics you feel
      useful.
      7. (Bonus-20 points) You may do some pre-processing on textual data to improve
      your classifier. Explain why score has improved if it did.
      8. Link to the dataset has been provided in the common link.
      9. You can use inbuilt functions for SVM.SMAI (CSE/ECE 471)

Q8. Clustering

      1. Given a dataset of documents with content from 5 different fields ( namely busi-
      ness, entertainment, politics, sport, and tech ), cluster them using any clustering
      algorithm of your choice.
      2. Do not use any libraries for this part. You are expected to code your clustering
      algorithm from scratch.
      3. For feature extraction you can use the vectorizers provided by sklearn or by using
      the pre trained embeddings. ( Code snippet for the usage of these embeddings has
      been provided in the previous question ).
      4. You might have to perform some pre-processing on the raw documents before you
      apply your algorithm.
      5. We have provided ground truth document tags for the documents. Report accuracy
      score on these documents.
      6. We will test your score on the documents for which the tags have not been provided.
      7. In the dataset, the number after the ’ ’ symbol in the file name denotes the cluster
      label.
      8. The code file must be a python(.py) file. You are expected to define a class for each
      question which is compatible with the test.py file provided here. Make sure your
      code can be run by ”python test.py”. Double check this.

Q9. Image Classification

      1. Given CiFAR-10 dataset, implement a linear SVM classifier to predict the classes
      of the test images.
      2. Featurize the images as vectors that can be used for classification.
      3. Report your observations for different values of C. Explain the significance of C.
      4. Compare and contrast the classifier with the KNN classifier built in the previous
      assignment.
      5. Report accuracy score, F1-score, Confusion matrix and any other metrics you feel
      useful.
      6. Report the support vector images in each case.
      7. (Bonus-20 points) You may do some processing on the train set to improve your
      scores on linear SVM. Report your changes clearly.
      8. You can use inbuilt functions for SVM.
