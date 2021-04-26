# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:46:16 2021

@author: Andrew Hallett and Himani Vommi
"""

#PHASE IV

#Create df from training data:
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
#plt.figure()
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
#plt.figure()
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

#plt.figure()
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


#---------------------------------------------------------------------
#KEEP ONLY 7 FEATURERS + ID (Based on previous decision tree analysis)
#---------------------------------------------------------------------

selectedFeatures = ['id','UrlLengthNorm', 'InsecureForms', 'NumDashNorm', 
    'NumPercentNorm', 'HostnameLengthNorm', 'MissingTitle', 'IframeOrFrame', 'CLASS_LABEL']
for col in df.columns:
    if col not in selectedFeatures:
        df.drop(columns=col, inplace=True)

print(df.columns)


#---------------------------------------------------------------------
#                        BUILDING THE MODEL
#---------------------------------------------------------------------
df.set_index('id')
print(df.head(5))
Y=df['CLASS_LABEL'].to_numpy() #y-value is classification (target)
X=df[['UrlLengthNorm', 'InsecureForms', 'NumDashNorm', 
    'NumPercentNorm', 'HostnameLengthNorm', 'MissingTitle', 'IframeOrFrame']].to_numpy()
#*** id shouldn't be in features, but will be needed later

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import svm
 
kf=KFold(n_splits=2, random_state=0, shuffle=True) 
# ^^ modify these to adjust model accuracy vv
C_values = np.linspace(0.1, 100, 20, endpoint=True)
 
 
avg_auc_test=[]
avg_auc_train=[]
avg_f1_test=[]
avg_f1_train=[]
 
for c in C_values:
    #these arrays will store acu value for each fold in cross validation 
    auc_train=[]
    auc_test=[]
    f1_train=[]
    f1_test=[]
   
    for train_index, test_index in kf.split(X):
        #X and Y are numpy arrays. 
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = svm.SVC(C=c,kernel='rbf') #rbf is best, but this can also be adjusted if we want
        clf.fit(X_train, Y_train)
        Y_test_pre=clf.predict(X_test)
        Y_train_pre=clf.predict(X_train)
        auc_train.append(roc_auc_score(Y_train,Y_train_pre))
        auc_test.append(roc_auc_score(Y_test,Y_test_pre))
        f1_test.append(f1_score(Y_test,Y_test_pre,pos_label=0))
        f1_train.append(f1_score(Y_train,Y_train_pre,pos_label=0))
         
     
    avg_auc_test.append(np.mean(auc_test))
    avg_auc_train.append(np.mean(auc_train))
    avg_f1_test.append(np.mean(f1_test))
    avg_f1_train.append(np.mean(f1_train))
 
 
plt.figure(figsize=(10,4))
plt.plot(C_values,avg_auc_test,label='Testing Set')    
plt.plot(C_values,avg_auc_train,label='Training Set')  
plt.legend()
plt.xticks(C_values,rotation='vertical')
plt.grid(color='b', axis='x', linestyle='-.', linewidth=1,alpha=0.2)
plt.xlabel('C')
plt.ylabel('AUC')
 
plt.figure(figsize=(10,4))
plt.plot(C_values,avg_f1_test,label='Testing Set')    
plt.plot(C_values,avg_f1_train,label='Training Set')  
plt.legend()
plt.xticks(C_values,rotation='vertical')
plt.grid(color='b', axis='x', linestyle='-.', linewidth=1,alpha=0.2)
plt.xlabel('C')
plt.ylabel('F1')


