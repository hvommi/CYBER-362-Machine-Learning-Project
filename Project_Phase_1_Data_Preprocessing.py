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
plt.show()

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
plt.show()

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
plt.show()
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
X = df.iloc[:, 0:numFeatures] #problem - this still contains CLASS LABEL
X.drop(labels='CLASS_LABEL', axis=1, inplace=True)
print(X)
Y = df.iloc[:, targetValIndex]
print(Y)
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()
#--------------------------------------------------------------
#Use DBSCAN for clustering: 
#import Kmeans
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
#define the model
clustermodel = DBSCAN(min_samples=10)
#create clusters
dsc = clustermodel.fit(df.to_numpy())
pred_y = clustermodel.fit_predict(df)
df['test'] = pred_y
import seaborn as sns

#pair plot to show cluserting vs CLASS_LABEL
sns.pairplot(df, hue='CLASS_LABEL')    
    
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
