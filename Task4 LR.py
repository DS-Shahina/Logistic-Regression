import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Importing Data
df = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Logistic Regression/bank_data.csv", sep = ",")
df.columns
df.head(11)
df.describe()
df.info()
df.isna().sum() # no na values

df.columns = 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue_collar','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself_employed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown','y' #renaming so that no sapces is there otherwise error.
df = df[['y', 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue_collar','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself_employed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown']] # rearranging columns
#############################################################

# Model building 

from sklearn.linear_model import LogisticRegression

X = df.iloc[:,1:]
y = df[["y"]]

log_model = LogisticRegression()
log_model.fit(X, y)

#############################################################

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

#Model Predictions
y_pred = log_model.predict(X)
y_pred

#Testing Model Accuracy
# Confusion Matrix for the model accuracy
confusion_matrix(y, y_pred)

# The model accuracy is calculated by (a+d)/(a+b+c+d)
accuracy = (39165 + 1177)/(45211) 
accuracy #0.8923049700294177

print(classification_report(y,y_pred)) # accuracy = 0.89

# As accuracy = 0.8923049700294177, which is greater than 0.5; Thus [:,1] Threshold value>0.5=1 else [:,0] Threshold value<0.5=0 
log_model.predict_proba(X)[:,1]

# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(y,log_model.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc) #auc accuracy: 0.6017876828997866 - Average model, it is less than 0.8 

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
classifier = LogisticRegression(random_state = 0)
model1 = classifier.fit(X_train, y_train)
#Testing model
from sklearn.metrics import confusion_matrix, accuracy_score
y_predtest = classifier.predict(X_test)
print(confusion_matrix(y_test,y_predtest))
print(accuracy_score(y_test,y_predtest)) #Accuracy = 0.8886021822471247 = 88%


#Training model
y_predtrain = classifier.predict(X_train)
print(confusion_matrix(y_train,y_predtrain))
print(accuracy_score(y_train,y_predtrain)) #Accuracy = 0.8933864189338642 = 89%

# train and test accuracy is close enough so it is good model.











# import statsmodels.formula.api as sm
logit_model = sm.logit('y ~ default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = df).fit()

#AIC means Akaike's Information Criteria and BIC means Bayesian Information Criteria. It should be less
#summary
logit_model.summary2() # for AIC:632.2126, BIC:698.1915 
logit_model.summary()

pred = logit_model.predict(df.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df.naffairs, pred) #It gives us FPR, TPR for different thresholds(cutoff)
optimal_idx = np.argmax(tpr - fpr) # TP Should be maximum as compare to FP
optimal_threshold = thresholds[optimal_idx] #at that maximum value what is the threshold(cutoff)
optimal_threshold #0.2521571570135329

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
print("Area under the ROC curve : %f" % roc_auc) # 0.720880

# filling all the cells with zeroes
df["pred"] = np.zeros(601) # add new column "pred" with all zeros
# taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > optimal_threshold, "pred"] = 1 # if the value is greater than threshold value mark it as "1" otherwise "0"
# classification report
classification = classification_report(df["pred"], df["naffairs"])
classification # support=accuracy=0.69 


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

#summary
model.summary2() # for AIC  AIC:450.2720   , BIC:510.8759 
model.summary() 

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of naffairs
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (90 + 30)/(181) 
accuracy_test #0.6629834254143646

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test #0.699194847020934 - average model, it should greater than 0.8 


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = (213 + 74)/(420)
print(accuracy_train) #0.6833333333333333

# train and test accuracy is close enough so we can accept.
