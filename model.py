# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:50:21 2021

@author: HP
"""

#  Attrition project
#  Dataset : Data Set.xlsx

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from pickle import load
from pickle import dump
import flask
import pickle

#  Load the data
attr = pd.read_excel("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/Data Set.xlsx")

#  Exploratory Data Analysis (EDA)
attr.describe()

# Change column names
attr.columns = "EmpID","EmpName","Department","Salary","Designation","LastSalaryHike","TrainingHoursInLast1Year", \
               "TotalWorkExperience","YearsSinceLastPromotion","EducationLevel","AdaptabilityandInitiative", \
               "AttendanceandPunctuality","LeadershipSkills","QualityofDeliverables","SelfEvaluation", \
               "ImprovementArea","JudgementandDecisionMaking","TeamWorkSkills","Productivity", \
               "CustomerRelationSkills","JobKnowledge", "ReliabilityandDependability"

# Drop features that is not relevant
attr = attr.drop(['EmpID','EmpName','Department','Designation','EducationLevel','SelfEvaluation','ImprovementArea'], axis=1)

###### Convert the "TrainingHoursInLast1Year" column from object to datetime
attr["TrainingHoursInLast1Year"] = pd.to_datetime(attr['TrainingHoursInLast1Year'], errors='coerce')
attr.info()

# Separate the "TrainingHoursInLast1Year" column in separate date and time columns
attr['Year'] = attr['TrainingHoursInLast1Year'].dt.year
attr['Month'] = attr['TrainingHoursInLast1Year'].dt.month
attr['Day'] = attr['TrainingHoursInLast1Year'].dt.day
attr['Hour'] = attr['TrainingHoursInLast1Year'].dt.hour
attr['Minutes'] = attr['TrainingHoursInLast1Year'].dt.minute
attr.info()

# The relevant columns are the "Hour" and "Minutes" columns
attr['TrainingLast1YearMinutes'] = attr['Hour']*60 + attr['Minutes']
attr.info()

# Check for null values/missing values
attr.isnull().sum()

# Missing data heatmap
cols = attr.columns[:]
colours = ['#000099', '#ffff00']  # specify the colours - yellow is missing, blue is not missing
sns.heatmap(attr[cols].isnull(), cmap=sns.color_palette(colours))

#  Replace/impute missing values with mean in column "TrainingLast1YearMinutes" which is relevant
aveTH = attr['TrainingLast1YearMinutes'].mean()
print(aveTH)
aveTH = int(aveTH)
print(aveTH)
attr['TrainingLast1YearMinutes'] = attr['TrainingLast1YearMinutes'].fillna(aveTH)

#  Re-arrange column "TrainingLast1YearMinutes" (to replace column "TrainingHoursInLast1Year")
attr = attr[['Salary','LastSalaryHike','TrainingLast1YearMinutes', \
             'TotalWorkExperience','YearsSinceLastPromotion','AdaptabilityandInitiative', \
             'AttendanceandPunctuality','LeadershipSkills','QualityofDeliverables', \
             'JudgementandDecisionMaking','TeamWorkSkills','Productivity', \
             'CustomerRelationSkills','JobKnowledge','ReliabilityandDependability']]  # all columns

#  Visualization
#  Scatter plot
attr.plot(x="TotalWorkExperience", y="Salary", kind="scatter")   
    
#  Histogram
num_bins = 10
attr.hist(bins=num_bins, figsize=(40,30))


#  Data pre-processing
#  Check for correlation
corrMatrix = attr.corr()
print(corrMatrix)
sns.heatmap(corrMatrix, annot=True)   # All the features show poor correlation ie less than 1.0 so include all the features


# Normalization function
def norm_func(i): 
    x = (i-i.min()) / (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
attr_norm = norm_func(attr.iloc[:,:])    
attr_norm.describe()

# Normalization function
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#X = attr.iloc[:,:]
#attr_norm = scaler.fit_transform(X)



######### ANY MODELS  ###########



#### Hierarchical Clustering  ######
#  For creating dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(attr_norm, method="ward", metric="euclidean")

#  Dendrogram
plt.figure(figsize=(15,22));plt.title('Dendrogram : Attrition');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
    leaf_rotation=0,   # rotates the x axis labels
    leaf_font_size=10   # font size for the x axis labels
)

#  Apply AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=4, linkage='ward', affinity="euclidean").fit(attr_norm)
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

attr['Cluster']=cluster_labels   # creating a new column and assigning it to new column

attr = attr.iloc[:, [15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
attr.head(10)

#  Aggregate mean of each cluster
attr.iloc[:, [1,2,3,4,5]].groupby(attr.Cluster).mean()

#  creating a csv file
attr.to_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/attritiontrial.csv", encoding="utf-8")

import os
os.getcwd()


#  Cluster count
clstr = pd.read_excel("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/Data Set with cluster.xlsx")

clstr.columns = "Cluster","EmpID","EmpName","Department","Salary","Designation","LastSalaryHike", \
                "TrainingHoursInLast1Year","TotalWorkExperience","YearsSinceLastPromotion", \
                "EducationLevel","AdaptabilityandInitiative", "AttendanceandPunctuality", \
                "LeadershipSkills","QualityofDeliverables","SelfEvaluation","ImprovementArea", \
                "JudgementandDecisionMaking","TeamWorkSkills","Productivity","CustomerRelationSkills", \
                "JobKnowledge", "ReliabilityandDependability"
clstr.info()

clstr['Cluster'].value_counts()  # 0:129(36.9%), 1:74(21.1%), 2:58(16.6), 3:89(25.4%)   permanent, contract, temporary, left
sns.countplot(x='Cluster', data=clstr, palette='hls')

clstr1 = clstr[['Cluster']]

#  Combine the "Cluster" feature to the normalised dataframe
attr_norm_cluster = pd.concat([clstr['Cluster'], attr_norm], axis=1)
attr_norm_cluster = pd.concat([clstr1, attr_norm], axis=1)
attr_norm_cluster

#  creating a csv file
attr_norm_cluster.to_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/Model.csv", index=False, encoding="utf-8")


# Visualize the cluster
sns.countplot(x='Cluster', hue='Department', data=clstr)
sns.countplot(x='Cluster', hue='Designation', data=clstr)
sns.countplot(x='Cluster', hue='LastSalaryHike', data=clstr)
sns.countplot(x='Cluster', hue='AdaptabilityandInitiative', data=clstr)  
sns.countplot(x='Cluster', hue='AttendanceandPunctuality', data=clstr)   
sns.countplot(x='Cluster', hue='LeadershipSkills', data=clstr)  # clear correlation 2
sns.countplot(x='Cluster', hue='QualityofDeliverables', data=clstr)
sns.countplot(x='Cluster', hue='SelfEvaluation', data=clstr)
sns.countplot(x='Cluster', hue='ImprovementArea', data=clstr)
sns.countplot(x='Cluster', hue='JudgementandDecisionMaking', data=clstr)
sns.countplot(x='Cluster', hue='TeamWorkSkills', data=clstr)
sns.countplot(x='Cluster', hue='Productivity', data=clstr)  
sns.countplot(x='Cluster', hue='CustomerRelationSkills', data=clstr)  # clear correlation 3
sns.countplot(x='Cluster', hue='JobKnowledge', data=clstr) # clear correlation 1
sns.countplot(x='Cluster', hue='ReliabilityandDependability', data=clstr)  # clear correlation 4

####################################################################################################


####################################################################################################
#  Feature Selection
# Select features using Univariate Selection
#from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
attrdata = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/Model.csv")
X = attrdata.iloc[:, 1:]
y = attrdata['Cluster']

# Apply SelectKBest class to see feature importance
ordered_rank_features = SelectKBest(score_func=chi2, k=15)
ordered_feature = ordered_rank_features.fit(X,y)
modelscores = pd.DataFrame(ordered_feature.scores_, columns=["Score"])
modelcolumns = pd.DataFrame(X.columns)
features_rank = pd.concat([modelcolumns,modelscores], axis=1)             
features_rank.columns = ['Feature','Score']
features_rank
features_rank.nlargest(15,'Score')

# Convert to Excel file
key_features = pd.DataFrame(features_rank.nlargest(15,'Score'))
key_features.to_excel("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/KeyFeatures.xlsx", index=False, encoding="utf-8")
                    
# Plot graph
feat_importance = pd.Series(ordered_feature.scores_, index=X.columns)
feat_importance = feat_importance.nlargest(15).plot(kind='barh')

# Drop the features which have scores of less than 1 ie TrainingLast1YearMinutes, YearsSinceLastPromotion and Productivity
attrdata1 = attrdata.drop(['TrainingLast1YearMinutes','YearsSinceLastPromotion','Productivity'], axis=1)
attrdata1.to_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv", index=False, encoding="utf-8")


#  To re-load sc when predicting new input from user below ie. use sc when doing normalization 
#  with MinMaxScaler to new input data from user (with same columns) and to serialise to disk 
#  using pickle
attrdata1data1 = attr[['Salary','LastSalaryHike','TotalWorkExperience','AdaptabilityandInitiative','AttendanceandPunctuality','LeadershipSkills','QualityofDeliverables','JudgementandDecisionMaking','TeamWorkSkills','CustomerRelationSkills','JobKnowledge','ReliabilityandDependability']]
# Normalization function
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X1 = attrdata1data1.iloc[:,:]
attrdata1data1_norm = sc.fit_transform(X1)

# Saving model to disk
pickle.dump(sc, open('scaler.pkl','wb'))

####################################################################################################



####################################################################################################
#  Model with Logistic Regression 
#  Dataset : ModelFinal.csv

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:]
y = attrdata1['Cluster']

#  Splitting the data into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(max_iter=50,random_state=5)  #50
logistic_regression.fit(X_train, y_train)

# Prediction on Test data
y_pred_test = logistic_regression.predict(X_test)
pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predictions'])

# Test accuracy
test_acc1 = metrics.accuracy_score(y_test, y_pred_test)
test_acc1  # 77.14%

# Check for precision and recall
print(classification_report(y_test, y_pred_test, labels=[0,1,2,3]))

# Prediction on Train Data
y_pred_train = logistic_regression.predict(X_train)
pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predictions'])

# Train accuracy
train_acc1 = metrics.accuracy_score(y_train, y_pred_train)
train_acc1  # 82.86%

# Check for precision and recall
print(classification_report(y_train, y_pred_train, labels=[0,1,2,3]))


# Hyperparameter tuning by GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {'max_iter':[50,100]}
lr = LogisticRegression(random_state=10)
lr_grid = GridSearchCV(estimator=lr, param_grid=params, cv=10)
lr_grid.fit(X_train,y_train)

# Prediction on Test data
y_grid_predict_test = lr_grid.predict(X_test)
pd.crosstab(y_test, y_grid_predict_test, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_test, y_grid_predict_test, labels=[0,1,2,3]))

# Test accuracy
gs_test_acc1 = metrics.accuracy_score(y_test, y_grid_predict_test)
gs_test_acc1  # 77.14%

# Prediction on Train data
y_grid_predict_train = lr_grid.predict(X_train)
pd.crosstab(y_train, y_grid_predict_train, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_train, y_grid_predict_train, labels=[0,1,2,3]))

# Train accuracy
gs_train_acc1 = metrics.accuracy_score(y_train, y_grid_predict_train)
gs_train_acc1  # 0.8285

lr_grid.best_score_
lr_grid.best_params_

###################################################################################################


###################################################################################################
#  Model with Random Forest 
#  Dataset : ModelFinal.csv

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:]
y = attrdata1['Cluster']

#  Splitting the data into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=200, random_state=10)  
random_forest.fit(X_train, y_train)

# Prediction on Test data
y_pred_test = random_forest.predict(X_test)  # to predict new data
pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames= ['Predictions']) 

# Test accuracy
test_acc2 = accuracy_score(y_test, y_pred_test)   
test_acc2  # 73.33% 

# Check for precision and recall
print(classification_report(y_test, y_pred_test))

# Prediction on Train data
y_pred_train = random_forest.predict(X_train)
pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames= ['Predictions']) 

# Train accuracy 
train_acc2 = accuracy_score(y_train, y_pred_train)
train_acc2   # 100.00%

# Check for precision and recall
print(classification_report(y_train, y_pred_train))


# Hyperparameter tuning by GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {'n_estimators':[50,100]}
rf = RandomForestClassifier(random_state=10)
rf_grid = GridSearchCV(estimator=rf, param_grid=params, cv=10)
rf_grid.fit(X_train,y_train)

# Prediction on Test data
y_grid_predict_test = rf_grid.predict(X_test)
pd.crosstab(y_test, y_grid_predict_test, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_test, y_grid_predict_test, labels=[0,1,2,3]))

# Test accuracy
gs_test_acc2 = metrics.accuracy_score(y_test, y_grid_predict_test)
gs_test_acc2  # 75.24%

# Prediction on Train data
y_grid_predict_train = rf_grid.predict(X_train)
pd.crosstab(y_train, y_grid_predict_train, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_train, y_grid_predict_train, labels=[0,1,2,3]))

# Train accuracy
gs_train_acc2 = metrics.accuracy_score(y_train, y_grid_predict_train)
gs_train_acc2   # 1.0

rf_grid.best_score_
rf_grid.best_params_


###################################################################################################



###################################################################################################
#  Model with Support Vector Machine
#  Dataset : ModelFinal.csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:]
y = attrdata1['Cluster']

#  Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)

from sklearn.svm import SVC 
model_svm = SVC(kernel='linear', random_state=10)   
model_svm.fit(X_train,y_train)

# Prediction on Test data
y_pred_test = model_svm.predict(X_test)
pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames= ['Predictions']) 

# Test accuracy
test_acc3 = accuracy_score(y_test, y_pred_test)   
test_acc3  # 77.14%%

# Check for precision and recall
print(classification_report(y_test, y_pred_test))

# Prediction on Train Data
y_pred_train = model_svm.predict(X_train)
pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predictions'])

# Train accuracy
train_acc3 = metrics.accuracy_score(y_train, y_pred_train)
train_acc3  # 83.26%

# Check for precision and recall
print(classification_report(y_train, y_pred_train))


# Hyperparameter tuning by GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {'C':[1,10,20], 'kernel':['rbf','linear']}

supportvm = SVC(random_state=10)
svm_grid = GridSearchCV(estimator=supportvm, param_grid=params, cv=10)
svm_grid.fit(X_train,y_train)

# Saving model to disk
pickle.dump(svm_grid, open('model.pkl','wb'))

# Prediction on Test data
y_grid_predict_test = svm_grid.predict(X_test)
pd.crosstab(y_test, y_grid_predict_test, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_test, y_grid_predict_test, labels=[0,1,2,3]))

# Test accuracy
gs_test_acc3 = metrics.accuracy_score(y_test, y_grid_predict_test)
gs_test_acc3  # 81.90%

# Print the training score of the best model
svm_grid.cv_results_
df = pd.DataFrame(svm_grid.cv_results_)[['param_C','param_kernel','mean_test_score']]

# Prediction on Train data
y_grid_predict_train = svm_grid.predict(X_train)
pd.crosstab(y_train, y_grid_predict_train, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_train, y_grid_predict_train, labels=[0,1,2,3]))

# Train accuracy
gs_train_acc3 = metrics.accuracy_score(y_train, y_grid_predict_train)
gs_train_acc3   # 95.92%

svm_grid.best_score_
svm_grid.best_params_

###############################################################################################





###############################################################################################
#  Model with Decision Tree
#  Dataset : ModelFinal.csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:]
y = attrdata1['Cluster']

#  Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)

from sklearn.tree import DecisionTreeClassifier as DT
dt = DT(criterion="entropy", random_state=10)
dt.fit(X_train,y_train)

# Prediction on Test data
y_pred_test = dt.predict(X_test)
pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predictions'])

# Test accuracy
test_acc4 = accuracy_score(y_test, y_pred_test)   
test_acc4  # 59.05%

# Check for precision and recall
print(classification_report(y_test, y_pred_test))

# Prediction on Train data
y_pred_train = dt.predict(X_train)
pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames= ['Predictions']) 

# Train accuracy 
train_acc4 = accuracy_score(y_train, y_pred_train)
train_acc4   # 1.0

# Check for precision and recall
print(classification_report(y_train, y_pred_train))


# Hyperparameter tuning by GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {'criterion':['entropy','gini']}
dtree = DT(random_state=10)
dt_grid = GridSearchCV(estimator=dtree, param_grid=params, cv=10)
dt_grid.fit(X_train,y_train)

# Prediction on Test data
y_grid_predict_test = dt_grid.predict(X_test)
pd.crosstab(y_test, y_grid_predict_test, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_test, y_grid_predict_test, labels=[0,1,2,3]))

# Test accuracy
gs_test_acc4 = metrics.accuracy_score(y_test, y_grid_predict_test)
gs_test_acc4  # 59.04%

# Prediction on Train data
y_grid_predict_train = dt_grid.predict(X_train)
pd.crosstab(y_train, y_grid_predict_train, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_train, y_grid_predict_train, labels=[0,1,2,3]))
help(DT)
# Train accuracy
gs_train_acc4 = metrics.accuracy_score(y_train, y_grid_predict_train)
gs_train_acc4   # 1.0

dt_grid.best_score_
dt_grid.best_params_



from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=10)
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)


###############################################################################################






###############################################################################################
#  Model with Neural Network
#  Dataset : ModelFinal.xlsx

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  #, Activation,Layer,Lambda

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:]
y = attrdata1['Cluster']

#  Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)

cont_model = Sequential()
cont_model.add(Dense(50, input_dim=12, activation="relu"))  # input_dim = no. of columns
cont_model.add(Dense(250, activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))  # plain vanilla model, assigning weights randomly
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model
model.fit(np.array(X_train), np.array(y_train), epochs=20)

# Prediction on Test data
y_pred_test = model.predict(np.array(X_test))
y_pred_test = pd.Series([i[0] for i in y_pred_test])

# Test accuracy
test_acc5 = np.corrcoef(y_pred_test, y_test)   
test_acc5  # 0.6369
test_acc5 = "0.6369"  # 71.50%

#  Prediction on Train data
y_pred_train = model.predict(np.array(X_train))
y_pred_train = pd.Series([i[0] for i in y_pred_train])

train_acc5 = np.corrcoef(y_pred_train, y_train)  
train_acc5  # 0.7913
train_acc5 = "0.7913"  # 86.74% this is just because some model's count the input layer and others don't

layerCount = len(model.layers)
layerCount  # 3

hiddenLayer = layerCount - 1;
lastLayer = layerCount - 2;

# getting the weights
hiddenWeights = model.layers[hiddenLayer].get_weights()
lastWeights = model.layers[lastLayer].get_weights()



# Hyperparameter tuning by GridSearchCV
#import required libraries
import pandas as pd #to manipulate data
import numpy as np #linear algebra
from sklearn.model_selection import train_test_split #to split data into train and test set
from sklearn.preprocessing import StandardScaler #to scale data
from keras.models import Sequential #to define model
from keras.layers import Dense #to add layers into model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV #to find best parameters

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:].values  # same as X=np.array(attrdata1.iloc[:, 1:])
y = attrdata1['Cluster'].values   # same as y=np.array(attrdata1.iloc[:, 0])

#  Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)

def build_model(optimizer):
    clf = Sequential()
    clf.add(Dense(units=18, activation='relu', input_dim=12))
    clf.add(Dense(units=18, activation='relu'))
    clf.add(Dense(units=1, activation='sigmoid'))
    clf.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the object of KerasClassifier class
clf = KerasClassifier(build_fn=build_model)

# Create the dictionary of the parameters
parameters = {'batch_size': [10, 25, 32],
              'epochs': [25, 100],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = clf, param_grid=parameters, scoring='accuracy', cv=10)

# Fit the model
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best parameters: ", best_parameters)
print("Best score: ", best_accuracy)   # 56.27%

    
###############################################################################################




###############################################################################################
#  Model with K-Nearest Neighbor
#  Dataset : ModelFinal.csv

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

attrdata1 = pd.read_csv("C:/Users/HP/Documents/My Documents 1/Cert in Data Science Course/Attrition Project/NS/trialtrialtrial/Data/ModelFinal.csv")
X = attrdata1.iloc[:, 1:]
y = attrdata1['Cluster']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=10)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Prediction on Test data
y_pred_test = knn.predict(X_test)
pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predictions'])

# Test accuracy
test_acc6 = accuracy_score(y_test, y_pred_test)   
test_acc6  # 74.29%

# Check for precision and recall
print(classification_report(y_test, y_pred_test))

# Prediction on Train data
y_pred_train = knn.predict(X_train)
pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames= ['Predictions']) 

# Train accuracy 
train_acc6 = accuracy_score(y_train, y_pred_train)
train_acc6   # 88.57%

# Check for precision and recall
print(classification_report(y_train, y_pred_train))


# Hyperparameter tuning by GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[5,10,15,19,21,30]}
knns = KNeighborsClassifier()
knn_grid = GridSearchCV(estimator=knns, param_grid=params, cv=10)
knn_grid.fit(X_train,y_train)

# Print the training score of the best model
knn_grid.cv_results_
df = pd.DataFrame(knn_grid.cv_results_)[['param_n_neighbors','mean_test_score']]
df
knn_grid.best_score_
knn_grid.best_params_

# Prediction on Test data
y_grid_predict_test = knn_grid.predict(X_test)
pd.crosstab(y_test, y_grid_predict_test, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_test, y_grid_predict_test, labels=[0,1,2,3]))

# Test accuracy
gs_test_acc6 = metrics.accuracy_score(y_test, y_grid_predict_test)
gs_test_acc6  # 75.24%

# Prediction on Train data
y_grid_predict_train = knn_grid.predict(X_train)
pd.crosstab(y_train, y_grid_predict_train, rownames=['Actual'], colnames=['Predictions'])
print(classification_report(y_train, y_grid_predict_train, labels=[0,1,2,3]))

# Train accuracy
gs_train_acc6 = metrics.accuracy_score(y_train, y_grid_predict_train)
gs_train_acc6   # 83.26%

knn_grid.best_score_
knn_grid.best_params_


###############################################################################################



#########################################################################################
# Select Support Vector Machine as best model
    

##########################################################################################

