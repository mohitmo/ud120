#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from collections import Counter

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC
clf = SVC(C=10000,kernel="rbf")
features_train = features_train[:len(features_train)/1000]
labels_train = labels_train[:len(labels_train)/1000]
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0,3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0,3), "s"

print "accuracy:", accuracy_score(labels_test, pred)
print "element 10:", pred[10]
print "element 26:", pred[26]
print "element 50:", pred[50]
c = Counter(pred) 
print "no. of sara mails", c[0]
print "no. of chris mails", c[1]
