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


def walk(dir):
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            tabl(path)
        else:
            walk(path)



def tabl(path):
    k = 0
    ids = []
    num_events = []
    f = open(path, 'r')
    temp_data = pd.DataFrame({
    'date_1': [],
    'time_1': [],
    'id': [],
    'is_online': [],
    'date_2': [],
    'time_2': []})
    for line in f:
        l = line.strip()
        temp_data.iloc[k].df.date_1 = l[]
        k=k+1

    f.close()
    print(temp_data)



s = '\ddd'
s1 = s[0]
directory = input("Напиши директорию: ")
walk(directory)
