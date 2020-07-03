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


#def walk(dir):
#    for name in os.listdir(dir):
#        path = os.path.join(dir, name)
#        if os.path.isfile(path):
#            tabl(path)
#        else:
#            walk(path)
#
#def tabl(path):



#def write_on_file(path):
#    f = open('path', 'r')
#    for line in f:
#        l = [line.strip()]
#        l
#    f.close()

path = 'D:\Test\users_data'
#directory = input("Напиши директорию: ")
#direct = walk(directory)
ids = []
num_events = []
for dir, subdir, files in os.walk(path):
    if len(dir.split('/')) > 1:
        ids.append(dir.split('/')[-1])
        num_events.append(len(files))


users_data = pd.DataFrame(columns=['num_events'], index=ids)
users_data.head()
users_data['num_events'] = num_events
users_data['online_freq'] = 0
users_data['id'] = users_data.index
users_data.describe()


online_freq_users = []
for dir, subdir, files in os.walk(path):
    if len(dir.split('\ '))>1:
        id = (dir.split('/')[-1])
        online_freq = []
        for f in files:
            temp_data = pd.read_table(dir+'/'+f, sep='|', names=
                                      ['date_1', 'time_1', 'id', 'is_online', 'date_2', 'time_2'])
            print(online_freq)
            freq = sum(temp_data['is_online'])/len(temp_data['is_online'])
            if freq.isna():
                online_freq.append()
                users_data.loc[users_data['id']==id, 'online_freq'] = sum(online_freq)

