# ML-Project-using-RandomForest
dataset consists of 70000 rows and 13 columns


Columns(features)


id           :-ID number


age          :-in days


gender       :- 1 - women, 2 - men


height       :-cm


weight       :-kg


ap_hi        :-Systolic blood pressure


ap_lo        :-Diastolic blood pressure


cholesterol  :- 1: normal, 2: above normal, 3: well above normal


gluc         :- 1:normal, 2: above normal, 3: well above normal


smoke        :-whether patient smokes or not


alco         :-Binary feature


active       :-Binary feature


cardio       :-Target variable


here i am using Random Forest algorithm to predict whether a person has cardio disease or not


Pros and cons of random forests

The advantages of random forests include:

The predictive performance can compete with the best supervised learning algorithms
They provide a reliable feature importance estimate
They offer efficient estimates of the test error without incurring the cost of repeated model training associated with cross-validation
On the other hand, random forests also have a few disadvantages:

An ensemble model is inherently less interpretable than an individual decision tree
Training a large number of deep trees can have high computational costs (but can be parallelized) and use a lot of memory
Predictions are slower, which may create challenges for applications ...
