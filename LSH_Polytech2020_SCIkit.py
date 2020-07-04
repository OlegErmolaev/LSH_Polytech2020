import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

users_data = pd.DataFrame()#users_data for predicting
users_data_train = pd.DataFrame()#user_data for training

model_tree_age = DecisionTreeClassifier(random_state=42, max_depth=3)
model_tree_age.fit(users_data_train[['online_freq_min','online_freq_max','freq_for_daytime', \
    'freq_for_morning', 'freq_for_evening', 'freq_for_night', 'maxonlinetimemean', \
    'maxofflinetimemean', 'onlinetimemax_for_day', 'onlinetime_for_morning', 'onlinetime_for_daytime', \
    'onlinetime_for_evening', 'onlinetime_for_night']], users_data_train['age']) #train our tree on parameters and age results
age_predict = model_tree_age.predict(users_data)
print(accuracy_score(user_data['age'], age_predict))

model_tree_sex = DecisionTreeClassifier(random_state=42, max_depth=3)
model_tree_sex.fit(users_data_train[['online_freq_min','online_freq_max','freq_for_daytime', \
    'freq_for_morning', 'freq_for_evening', 'freq_for_night', 'maxonlinetimemean', \
    'maxofflinetimemean', 'onlinetimemax_for_day', 'onlinetime_for_morning', 'onlinetime_for_daytime', \
    'onlinetime_for_evening', 'onlinetime_for_night']], users_data_train['sex']) #train our tree on parameters and sex results
sex_predict = model_tree_age.predict(users_data)
print(accuracy_score(user_data['sex'], sex_predict))