# Austin-animal-shelter-outcomes

The Austin Animal Center is the largest no-kill animal shelter in the United States that provides care and shelter to over 18,000 animals each year and is involved in a range of county, city, and state-wide initiatives for the protection and care of abandoned, at-risk, and surrendered animals.
As part of the City of Austin Open Data Initiative, the Austin Animal Center makes available its collected dataset that contains statistics and outcomes of animals entering the Austin Animal Services system. 
This project’s goal is to predict the adoption outcome of animals entering the Austin Animal Center. 
This dataset is publicly available on Kaggle. The objective is to predict if an animal will get adopted or not based on it’s type (cat/dog), breed, sex, age, color etc. The animals (cats and dogs) belong to Austin animal shelter. 
Kaggle link - https://www.kaggle.com/aaronschlegel/austin-animal-center-shelter-outcomes-and
The problem is a typical binary classification problem. We would be building classifiers that use ensemble methods in this project. Our main evaluation metric would be the classification accuracy. 

### Libraries used

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

import pickle

np.random.seed(42)

import time

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
