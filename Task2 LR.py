import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Logistic Regression/advertising.csv", sep = ",")
df.columns

#removing 'Ad_Topic_Line','Timestamp','City' and 'Country' column because it is not necessary
df = df.drop(['Ad_Topic_Line','Timestamp','City','Country'], axis = 1)

df.head(11)
df.describe()
df.isna().sum() #No na values
df.columns

df.columns = 'Daily_Time_Spent_on_Site', 'Age', 'Area_Income','Daily_Internet_Usage', 'Male', 'Clicked_on_Ad' #renaming so that no sapces is there otherwise error.
df = df[['Clicked_on_Ad', 'Daily_Time_Spent_on_Site', 'Age', 'Area_Income','Daily_Internet_Usage', 'Male']] # rearranging columns
#############################################################

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = df).fit()

#AIC means Akaike's Information Criteria and BIC means Bayesian Information Criteria. It should be less
#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(df.iloc[ :, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df.Clicked_on_Ad, pred) #It gives us FPR, TPR for different thresholds(cutoff)
optimal_idx = np.argmax(tpr - fpr) # TP Should be maximum as compare to FP
optimal_threshold = thresholds[optimal_idx] #at that maximum value what is the threshold(cutoff)
optimal_threshold #0.6326111998951834

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red') #True Positive Rate - Sensitivity
pl.plot(roc['1-fpr'], color = 'blue') # True Negative Rate
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc) # 0.991800

# filling all the cells with zeroes
df["pred"] = np.zeros(1000) # add new column "pred" with all zeros
# taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > optimal_threshold, "pred"] = 1 # if the value is greater than threshold value mark it as "1" otherwise "0"
# classification report
classification = classification_report(df["pred"], df["Clicked_on_Ad"])
classification # accuracy=0.97 


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = train_data).fit()

#summary
model.summary2() # for AIC  AIC:137.6533  , BIC:164.9598
model.summary() 

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = (149 + 142)/(300) 
accuracy_test #0.97

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test #0.9978620106008641 - good model, it is greater than 0.8 

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = (346 + 335)/(700)
print(accuracy_train) #0.9728571428571429

# train and test accuracy is same so it is good model.

