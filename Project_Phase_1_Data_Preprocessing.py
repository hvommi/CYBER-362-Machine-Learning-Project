# -*- coding: utf-8 -*-

"""

Created on Sun Mar 21 23:47:46 2021



@author: Andrew Hallett and Himani Vommi

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer



#df=pd.read_csv('C:/Users/Andrew/Desktop/Cyber 362 Security Analytics Studio/Phishing_Legitimate_train_missing_data.csv',na_values=['',' ','n/a'])
df=pd.read_csv('Phishing_Legitimate_train_missing_data.csv',na_values=['',' ','n/a'])
#setting index to id

df.set_index('id')
print(df.head(4))

#Checking sums of missing values for each column
df.isna().sum()

#true or false to show what rows have missing values

#print(df.isnull().any(axis=1))

#checking rows for missing values

rows_withna=df[ df.isnull().any(axis=1)]
print(df.shape)
print("rows with N/A: ")
print(rows_withna)

#Checking the index of each row with missing values 

rows_withnas_index= df[ df.isnull().sum(axis=1) >= 1 ].index

print("indexes where n/a located:")
print(rows_withnas_index)
#3O rows contain NaN values
print("--------------------------------------------------------------------")

#define an imputer that will replace missing values for the 30 rows with missing values

imputer = KNNImputer(n_neighbors=30)#create numpy array of columns

#continuous variables only: 
to_be_replaced=df[['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents',
                   'NumAmpersand','NumHash','NumNumericChars','HostnameLength','PathLength','QueryLength','NumSensitiveWords',
                   'PctExtResourceUrls']].to_numpy()

#listed in canvas table but not in CSV: 
#'PctExtHyperlinks','SubdomainLevelRT', 'UrlLengthRT', 'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'

to_be_replaced_imputed = imputer.fit_transform(to_be_replaced)

print(to_be_replaced_imputed)

#assign the imputed values to the dataframe

df[['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents',
                   'NumAmpersand','NumHash','NumNumericChars','HostnameLength','PathLength','QueryLength','NumSensitiveWords',
                   'PctExtResourceUrls']]=to_be_replaced_imputed

print(df)


#confirm no more missing values left after imputing:
missing_df = df[df.isna().any(axis=1)]
print("rows with missing values AFTER IMPUTING: ")
print(missing_df)

rows_withnas_index= df[ df.isnull().sum(axis=1) >= 1 ].index
print("remaining indexes where n/a located:")
print(rows_withnas_index)
#Removing each index of remaining rows with NaN values
print("Shape before dropping remaining N/A rows: " + str(df.shape))
df.drop(rows_withnas_index, inplace=True)
#Printing the new shape of the df with the rows dropped
print("Shape after dropping N/A rows: " + str(df.shape))
#----------------------------------------------------------------------
#determining outliers:
#in the interest of readability, only a few columns' data will be shown on the box plot to demonstrate outliers:
plt.figure()
df[['NumDots', 'PathLevel', 'UrlLength', 'NumDash']].plot.box()
#plt.show()

#print(df.columns)
#----------------------------------------------------------------------
#removing outliers with negative outlier factor less than -1.5:
from sklearn.neighbors import LocalOutlierFactor
#check 100 neighbors
clf = LocalOutlierFactor(n_neighbors=100)
X=df[['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents',
                   'NumAmpersand','NumHash','NumNumericChars','HostnameLength','PathLength','QueryLength','NumSensitiveWords',
                   'PctExtResourceUrls']].to_numpy()
#find the label of outliers 
outlier_label=clf.fit_predict(X)
print(clf.negative_outlier_factor_)
#print(clf.offset_)
print(outlier_label)

#identify the index of the rows to drop 
print(df.shape)
rows_to_drop= df.iloc[ clf.negative_outlier_factor_ < -1.5].index
print("Outlier rows to drop: \n")
print(rows_to_drop)
df.drop(rows_to_drop,inplace=True)
print(df.shape)

#box plot still has additional outlier dots, but the shape was notably updated:
plt.figure()
df[['NumDots', 'PathLevel', 'UrlLength', 'NumDash']].plot.box()
#plt.show()

#------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
x=df[['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents',
                   'NumAmpersand','NumHash','NumNumericChars','HostnameLength','PathLength','QueryLength','NumSensitiveWords',
                   'PctExtResourceUrls']].to_numpy()
scaler=StandardScaler() 
#plt.boxplot(x) 
scaler.fit(x)
print(scaler.mean_)
print(scaler.var_)
#scale the x values
x=scaler.transform(x)
df[['NumDotsNorm','SubdomainLevelNorm','PathLevelNorm','UrlLengthNorm','NumDashNorm','NumDashInHostnameNorm','NumUnderscoreNorm',
                    'NumPercentNorm','NumQueryComponentsNorm','NumAmpersandNorm','NumHashNorm','NumNumericCharsNorm','HostnameLengthNorm',
                    'PathLengthNorm','QueryLengthNorm','NumSensitiveWordsNorm','PctExtResourceUrlsNorm']]=x

plt.figure()
df[['NumDotsNorm', 'PathLevelNorm', 'UrlLengthNorm', 'NumDashNorm']].plot.box()
#plt.show()
print("Shape after normalizing data: " + str(df.shape))
#------------------------------------------------------------------------
#clean up old columns:
df.drop(columns=['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents',
                   'NumAmpersand','NumHash','NumNumericChars','HostnameLength','PathLength','QueryLength','NumSensitiveWords',
                   'PctExtResourceUrls'], inplace=True)
print(df.columns)
print("Shape after final clean up of data: " + str(df.shape))

#------------------------------------------------------------
#   PHASE II-III: DATA EXPLORATION
#------------------------------------------------------------
#Removing featueres that are 80% correlated:
    
correlated_features = set() #for storing correlated values
correlation_matrix = df.corr()
column_pairs = [] #correlation value for each column pair added from matrix

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.70:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname) 
            column_pairs.append(str(correlation_matrix.columns[i]) + " vs. " 
                                + str(correlation_matrix.columns[j]) + ": "
                             + str(abs(correlation_matrix.iloc[i, j]) ))
df.drop(labels=correlated_features, axis=1, inplace=True)
print("Shape of data after dropping correlated features: " + str(df.shape))
#-----------------------------------------------------------
print(correlated_features)
print(column_pairs)
#Removing features with very little variety:
# from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold=0.01) #threshold for similarity 
constant_filter.fit(df)
len(df.columns[constant_filter.get_support()])
constant_columns = [column for column in df.columns
 if column not in df.columns[constant_filter.get_support()]]
print(constant_columns) #prints the featurers in dataset that contain constant values 0 no varieety
df.drop(labels=constant_columns, axis=1, inplace=True)
print("Shape of data after dropping features with little variation: " + str(df.shape))

#--------------------------------------------------------------
#Setting up heat map to identify variables that have strongest correlation with target:
import seaborn as sns
features = []
index = 0
targetValIndex = -1
for column in df.columns:
    if column != 'CLASS_LABEL':
        features.append(column)
    else:
       targetValIndex = index #index of class label
    index += 1

X = features 
Y =['CLASS_LABEL'] #target column = CLASS LABEL 
# print(df.columns)

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
 
#--------------------------------------------------------------
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
numFeatures = len(features)
X = df.iloc[:, 0:numFeatures]  
X.drop(labels='CLASS_LABEL', axis=1, inplace=True)
print("df x" + str(X))
Y = df.iloc[:, targetValIndex]
print("df y: " + str(Y))
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(7).plot(kind='barh')
#plt.show()
#--------------------------------------------------------------
#Use DBSCAN for clustering: 
#import Kmeans
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# #define the model
# clustermodel = DBSCAN(min_samples=10)
# #create clusters
# dsc = clustermodel.fit(df.to_numpy())
# pred_y = clustermodel.fit_predict(df)
# df['test'] = pred_y
# import seaborn as sns

#pair plot to show cluserting vs CLASS_LABEL
#sns.pairplot(df, hue='CLASS_LABEL')    

#------------------------------------------------------------
#   PHASE IV: Exploratory Data Analysis & Feature selection Using Decision Trees
#------------------------------------------------------------

#Creating Decision Tree
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

print("decision trees??")
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X, Y)
#plt.figure(figsize=(70,70))
tree.plot_tree(clf.fit(X, Y), filled=True, fontsize=8)
    
#k-fold + assessing min_sample_split value: 
kf=KFold(n_splits=5, random_state=None, shuffle=True)
 
#Return evenly spaced numbers over a specified interval.
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
print(min_samples_splits)
 
#avg_f1_test=[]
avg_f1_train=[]
#this array will be used to display the number of the trees
avg_n_leaves=[]
#try different min sample splits
#these arrays will store f1 value for each fold in cross validation 
f1_train=[]
#f1_test=[]
recall_train=[]
avg_recall_train=[]
precision_train=[]
avg_precision_train=[]
auc_train=[]
avg_auc_train=[]
for mss in min_samples_splits:
     
   
     
    #perform k-fold for the given min_sample_split value
    for train_index, test_index in kf.split(X):
        #X and Y are dataframes. Therefore, we use iloc to select rows.
        X_train =X.iloc[train_index] 
        Y_train =Y.iloc[train_index] 
      
        #create a new model for the given min_samples_split
        clf = tree.DecisionTreeClassifier(min_samples_split=mss, max_depth=3)
        #use the training set to train the tree
        clf = clf.fit(X_train, Y_train)
        #predict both training and testing cases. 
        #Y_test_predicted=clf.predict(X_test)
        Y_train_predicted=clf.predict(X_train)
         
        #store the f1 scores of the training and testing sets.  
        #f1_test.append(f1_score(Y_test,Y_test_predicted,pos_label=0))       
        
        f1_train.append(f1_score(Y_train,Y_train_predicted,pos_label=0))
        auc_train.append(roc_auc_score(Y_train, Y_train_predicted))
        recall_train.append(recall_score(Y_train, Y_train_predicted))
        precision_train.append(precision_score(Y_train, Y_train_predicted))
        #n_leaves.append(clf.get_n_leaves())
     
    #calculate the average of the f1 scores after cross validation     
    #avg_f1_test.append(np.mean(f1_test))
    avg_f1_train.append(np.mean(f1_train))   
    avg_auc_train.append(np.mean(auc_train))
    avg_recall_train.append(np.mean(recall_train))
    avg_precision_train.append(np.mean(precision_train))
                            
    #avg_n_leaves.append(np.mean(n_leaves))
     
for x in range(0, len(avg_auc_train)):
    print(" | AUC: " + str(avg_auc_train[x]) + " | Recall: " + str(avg_recall_train[x]) 
          + " | Precision: " + str(avg_precision_train[x]) + " | F1: " + str(avg_f1_train[x]))
    
#     #------ min split vs. f1
# plt.figure(figsize=(4,4))
# #plt.plot(min_samples_splits,avg_f1_test,label='Testing Set')    
# plt.plot(min_samples_splits,avg_f1_train,label='Training Set')  
# plt.legend()
# plt.xticks(min_samples_splits)
# plt.grid(color='b', axis='x', linestyle='-.', linewidth=1,alpha=0.2)
# plt.xlabel('Minimum Sample Split Fraction')
# plt.ylabel('F1')

#--------- min split vs. auc curve
plt.figure(figsize=(4,4))
#plt.plot(min_samples_splits,avg_f1_test,label='Testing Set')    
plt.plot(min_samples_splits,avg_auc_train,label='Training Set')   
plt.legend()
plt.xticks(min_samples_splits)
plt.grid(color='b', axis='x', linestyle='-.', linewidth=1,alpha=0.2)
plt.xlabel('Minimum Sample Split Fraction')
plt.ylabel('AUC')

plot_roc_curve(clf, X, Y)

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# clf = tree.DecisionTreeClassifier(min_samples_split=0.1)  #based on results from previous run above
# clf = clf.fit(X, Y)
# #plt.figure(figsize=(70,70))
# tree.plot_tree(clf.fit(X, Y), filled=True, fontsize=8)
    
#k-fold + assessing min_sample_split value: 
kf=KFold(n_splits=5, random_state=None, shuffle=True)
 
#Return evenly spaced numbers over a specified interval.
max_depths = np.linspace(1, 29, 11, endpoint=True)
print(max_depths)
 
#avg_f1_test=[]
avg_f1_train=[]
#this array will be used to display the number of the trees
avg_n_leaves=[]
#try different min sample splits
#these arrays will store f1 value for each fold in cross validation 
f1_train=[]
#f1_test=[]
recall_train=[]
avg_recall_train=[]
precision_train=[]
avg_precision_train=[]
auc_train=[]
avg_auc_train=[]
for md in max_depths:
     
   
     
    #perform k-fold for the given min_sample_split value
    for train_index, test_index in kf.split(X):
        #X and Y are dataframes. Therefore, we use iloc to select rows.
        X_train =X.iloc[train_index] 
        Y_train =Y.iloc[train_index] 
      
        #create a new model for the given min_samples_split
        clf = tree.DecisionTreeClassifier(min_samples_split=0.1, max_depth=md)
        #use the training set to train the tree
        clf = clf.fit(X_train, Y_train)
        #predict both training and testing cases. 
        #Y_test_predicted=clf.predict(X_test)
        Y_train_predicted=clf.predict(X_train)
         
        #store the f1 scores of the training and testing sets.  
        #f1_test.append(f1_score(Y_test,Y_test_predicted,pos_label=0))       
        auc_train.append(roc_auc_score(Y_train, Y_train_predicted))
        f1_train.append(f1_score(Y_train,Y_train_predicted,pos_label=0))
        recall_train.append(recall_score(Y_train, Y_train_predicted))
        precision_train.append(precision_score(Y_train, Y_train_predicted))
        #n_leaves.append(clf.get_n_leaves())
     
    #calculate the average of the f1 scores after cross validation     
    #avg_f1_test.append(np.mean(f1_test))
    avg_f1_train.append(np.mean(f1_train))
    avg_auc_train.append(np.mean(auc_train))
    avg_recall_train.append(np.mean(recall_train))
    avg_precision_train.append(np.mean(precision_train))
                            
    #avg_n_leaves.append(np.mean(n_leaves))
     
for x in range(0, len(avg_auc_train)):
    print(" | AUC 2: " + str(avg_auc_train[x]) + " | Recall: " + str(avg_recall_train[x]) 
          + " | Precision: " + str(avg_precision_train[x]) + " | F1: " + str(avg_f1_train[x]))
    
#     #------ min split vs. f1
# plt.figure(figsize=(4,4))
# #plt.plot(min_samples_splits,avg_f1_test,label='Testing Set')    
# plt.plot(min_samples_splits,avg_f1_train,label='Training Set')  
# plt.legend()
# plt.xticks(min_samples_splits)
# plt.grid(color='b', axis='x', linestyle='-.', linewidth=1,alpha=0.2)
# plt.xlabel('Minimum Sample Split Fraction')
# plt.ylabel('F1')


# plt.figure(figsize=(4,4))
# tree.plot_tree(clf.fit(X, Y), filled=True, fontsize=8)
#plt.plot(min_samples_splits,avg_f1_test,label='Testing Set')
#--------- max depth vs. auc curve    
plt.plot(max_depths,avg_auc_train,label='Training Set')   
plt.legend()
plt.xticks(max_depths)
plt.grid(color='b', axis='x', linestyle='-.', linewidth=1,alpha=0.2)
plt.xlabel('Maximum Depth')
plt.ylabel('AUC')
plt.show()

plot_roc_curve(clf, X, Y)

 #----------------------------------------
#roc plots: 
#29 max depth vs 6 max depth w/ mss = 0.1
clf_29 = tree.DecisionTreeClassifier(min_samples_split=0.1, max_depth=29)
 #use the training set to train the tree
clf_29 = clf_29.fit(X, Y) #original x/y dataframes
plot_roc_curve(clf_29, X, Y)

clf_6 = tree.DecisionTreeClassifier(min_samples_split=0.1, max_depth=6)
 #use the training set to train the tree
clf_6 = clf_6.fit(X, Y) #original x/y dataframes
plot_roc_curve(clf_6, X, Y)

#--------------------------------------------------------------
# #use chi-squared values to identify seven most relevant features: 
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# #apply SelectKBest class to extract top 7 best features
# bestfeatures = SelectKBest(score_func=chi2, k=7)
# numFeatures = len(features)
# X = df.iloc[:, 0:numFeatures]
# print(X)
# Y = df.iloc[:, targetValIndex]
# fit = bestfeatures.fit(X,Y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score'] #naming the dataframe columns
# print(featureScores.nlargest(7,'Score')) #print 10 best features
