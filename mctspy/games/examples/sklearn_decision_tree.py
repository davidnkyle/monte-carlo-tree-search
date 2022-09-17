import glob
import os

import pandas as pd
import psutil
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle
import time

startTime = time.time()

folder_path = r'G:\Users\DavidK\analyses_not_project_specific\20220831_simulation_results\20220912_results/*.csv'
all_files = glob.glob(folder_path)
li = []

print('reading data')

for filename in all_files:
    df = pd.read_csv(filename, index_col='Unnamed: 0')
    li.append(df)
    print('.', end='')


df = pd.concat(li)
del li

print()
executionTime = (time.time() - startTime) / 60
print('Execution time in minutes: ' + str(executionTime))
print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

y_train = np.array(df['result'])
x_train = df.drop('result', axis=1).values

print('fitting model')

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)

print('writing results')

with open('model_3pl_rounds_left4.pkl', 'wb') as f:
    pickle.dump(model, f)

executionTime = (time.time() - startTime) / 60
print('Execution time in minutes: ' + str(executionTime))
print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

