######################loading the packages#########################
import pandas as pd
import os as os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import ttest_ind
import random
random.seed( 0)

#Introduction:
#For the present project the data from https://www.kaggle.com/keremdlkmn/student-alcohol-data-science/data
#was used. The data contains socioeconomical information of portuguesse students and some
#result variables G1, G2, G3, the last variable was used as target since that it was
#the final grade of portuguesse course. The features are compossed by some information related
#with social characteristics, use of time, student plans ann desires, parent caracteristics, etc.
    
##Setting the working directory
os.chdir("C:/Users/paco_/Documents/MASNA/Machine Learning/Final_Project")
##Loading the data
stud=pd.read_csv("data/student-por.csv",sep=';',header=0)
##Some basic data set information
stud.shape
stud.info()
stud.head()
##Total missing samples per column
stud[stud.columns].isnull().sum()
##Not missing values were founded in the data set
##Useful list for the analysis
continuous=['age','absences','G1','G2','G3']##Continuos variables

binary=['school','sex','address','famsize',
          'Pstatus','schoolsup',
          'famsup','paid','activities','nursery',
          'higher','internet','romantic']##Binary variables

nominal=['Mjob','Fjob','reason','guardian']##Discrete multinomial variables

ordinal=['Medu','Fedu','traveltime','studytime',
         'failures','famrel','freetime','goout',
         'Dalc','Walc','health'] ##Ordinal variables

stud[continuous].describe() ##Some descriptive statistics for continuos variables


#####Barplot for binary variables
for var in binary:
    stud[var].value_counts().plot(kind='bar',title='Student by '+var, color=['limegreen','powderblue'],edgecolor='k')
    plt.show()

#####Barplot for nominal variables
for var in nominal:
    stud[var].value_counts().plot(kind='bar',title='Student by '+var,edgecolor='k')
    plt.show()

#####Barplot for ordinal variables
for var in ordinal:
    stud[var].value_counts().plot(kind='bar',title='Student by '+var,edgecolor='k')
    plt.show()

#####Barplot for continuos variables
for var in continuous:
    plt.hist(stud[var],bins=10, facecolor='powderblue',normed=1,edgecolor='k')
    plt.title('Histogram '+var)
    plt.grid(True)
    plt.show()

####Boxplot for continuos variables
stud[continuous].plot(kind='box',subplots=True,layout=(3,2),sharex=False,sharey=False,figsize=(8,8))

######Scatter matrix for continuos variables##################
scatter_matrix(stud[continuous],figsize=(8,8),color='powderblue', edgecolor='k',alpha=0.2)

####Correlation between continuos variables
stud[continuous].corr()

####Pearson test for correlations 
print('Correlation test for continuous variables')
for i in range(0,len(continuous)):
    for j in range(i+1,len(continuous)):
        print('corr between', continuous[i] , 'and', continuous[j], ' : r=', round(pearsonr(stud[continuous[i]],stud[continuous[j]])[0],3),' p=',round(pearsonr(stud[continuous[i]],stud[continuous[j]])[1],3))

####T-test for G3 mean: binary features        
for var in binary:
    print('G3 mean analisys by '+ var)
    print(stud.groupby([var]).mean()['G3'])
    x=stud.where(stud[var]==np.unique(stud[var])[0]).dropna()['G3']
    y=stud.where(stud[var]==np.unique(stud[var])[1]).dropna()['G3']
    print(ttest_ind(x,y))
    plt.figure()
    sns.boxplot(x=var,y='G3',data=stud)
    plt.show()

#The null hypothesis is not rejected in the variables: 
#famsize, parents cohabitation, scholar support, family support,
#paid, extracurricular activities, and nursery.

####Spearman rho test between G3 and ordinal features        
for var in ordinal:
    print('Spearman rho test between G3 and' + ' ' + var )
    print(spearmanr(stud.G3,stud[var]))
    

#The null hypothesis of nulity correlation of correlation is not
#rejected in the variables: famrel
    
###Kruskal Wally test for nominal variables
from scipy.stats import kruskal
###########Kruskal Wallis test for Mother's job
v=stud.where(stud['Mjob']==np.unique(stud['Mjob'])[0]).dropna()['G3']
w=stud.where(stud['Mjob']==np.unique(stud['Mjob'])[1]).dropna()['G3']
x=stud.where(stud['Mjob']==np.unique(stud['Mjob'])[2]).dropna()['G3']
y=stud.where(stud['Mjob']==np.unique(stud['Mjob'])[3]).dropna()['G3']
z=stud.where(stud['Mjob']==np.unique(stud['Mjob'])[4]).dropna()['G3']
stat, p = kruskal(v, w, x, y, z)
print('Statistics=%.3f, p=%.3f' % (stat, p))
####The null hypothesis of equal distribution is rejected ####

###########Kruskal Wallis test for Father's job
v=stud.where(stud['Fjob']==np.unique(stud['Fjob'])[0]).dropna()['G3']
w=stud.where(stud['Fjob']==np.unique(stud['Fjob'])[1]).dropna()['G3']
x=stud.where(stud['Fjob']==np.unique(stud['Fjob'])[2]).dropna()['G3']
y=stud.where(stud['Fjob']==np.unique(stud['Fjob'])[3]).dropna()['G3']
z=stud.where(stud['Fjob']==np.unique(stud['Fjob'])[4]).dropna()['G3']
stat, p = kruskal(v, w, x, y, z)
print('Statistics=%.3f, p=%.3f' % (stat, p))
####The null hypothesis of equal distribution is rejected ####

###########Kruskal Wallis test for reason of choosing college
v=stud.where(stud['reason']==np.unique(stud['reason'])[0]).dropna()['G3']
w=v=stud.where(stud['reason']==np.unique(stud['reason'])[1]).dropna()['G3']
x=v=stud.where(stud['reason']==np.unique(stud['reason'])[2]).dropna()['G3']
y=v=stud.where(stud['reason']==np.unique(stud['reason'])[3]).dropna()['G3']
stat, p = kruskal(v, w, x, y)
print('Statistics=%.3f, p=%.3f' % (stat, p))
####The null hypothesis of equal distribution is rejected ####

###########Kruskal Wallis test for reason of choosing college
v=stud.where(stud['guardian']==np.unique(stud['guardian'])[0]).dropna()['G3']
w=stud.where(stud['guardian']==np.unique(stud['guardian'])[1]).dropna()['G3']
x=stud.where(stud['guardian']==np.unique(stud['guardian'])[2]).dropna()['G3']
stat, p = kruskal(v, w, x)
print('Statistics=%.3f, p=%.3f' % (stat, p))
####The null hypothesis of equal distribution is rejected ####
stud_1=stud.drop(['famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid',
'activities', 'nursery', 'famrel','G1','G2'], axis=1)


###################Encoding labels#############################
###Reconding variables for posterior analysis
stud_1['health'].columns='Health'
stud_1.rename(columns={'health':'Health'},inplace=True)
stud_1=pd.concat([stud_1.drop('Mjob',axis=1),pd.get_dummies(stud_1['Mjob'])],axis=1)
stud_1.rename(columns={'at_home':'M_at_home', 'health':'M_health', 'other':'M_other',
       'services':'M_services', 'teacher':'M_teacher'},inplace=True)
stud_1=pd.concat([stud_1.drop('Fjob',axis=1),pd.get_dummies(stud_1['Fjob'])],axis=1)
stud_1.rename(columns={'at_home':'F_at_home', 'health':'F_health', 'other':'F_other',
       'services':'F_services', 'teacher':'F_teacher'},inplace=True)
stud_1=pd.concat([stud_1.drop('reason',axis=1),pd.get_dummies(stud_1['reason'])],axis=1)
stud_1.rename(columns={'course':'r_course', 'home':'r_home', 'other':'r_other',
       'reputation':'r_reputation'},inplace=True)
stud_1=pd.concat([stud_1.drop('guardian',axis=1),pd.get_dummies(stud_1['guardian'])],axis=1)
stud_1.rename(columns={'father':'g_father', 'mother':'g_mother', 'other':'g_other'},inplace=True)

##################Preprocessing########################
######Considering the information found in https://www.iseg.ulisboa.pt/aquila/unidade/ERASMUS/incoming-mobility/academic-information/ects--european-credit-transfer-system
##### and https://www.iseg.ulisboa.pt/aquila/unidade/ERASMUS/incoming-mobility/academic-information/ects--european-credit-transfer-system
##### We recode the marks for transforming the task into a classfication problem considering the pass and non-pass marks
bin=[0,10,20]
Y = pd.cut(stud_1.G3,bin,right=False,labels=False)
Y[Y==1]=2
Y[Y==0]=1
Y[Y==2]=0

####Considering the previous analysis we choose the features to be included in the 
###machine learning task:
X = stud_1.drop(['G3'],axis=1)
X = pd.get_dummies(X, prefix=[column for column in X.columns if X[column].dtype == object], 
                   drop_first=True)

###We set the train and test features and labels########################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.333, random_state=10)
#######Standardization of train and test data
from sklearn.preprocessing import StandardScaler
scaler_train = StandardScaler().fit(X_train)
X_train = scaler_train.transform(X_train) 
X_test = scaler_train.transform(X_test)

########Import some metrics for evaluating the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import jaccard_similarity_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

import numpy as np

#taking from: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
####################Random forest model#############################
#######Defining a list of measures to be used in calibration of model##################
param_grid_1 = {
    'bootstrap': [True],
    'max_depth': [3, 4, 6, 8],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6, 10, 12],
    'n_estimators': [10, 20, 50, 100]
}

####Define the classifier
random_f=RandomForestClassifier(class_weight='balanced')
####Set the Grid search for getting the best combination of parameters for the
#random forest model
f1_m=make_scorer(f1_score,average='micro', greater_is_better=True)
js=make_scorer(jaccard_similarity_score,greater_is_better=True)

grid_search_1 = GridSearchCV(estimator = random_f, param_grid = param_grid_1, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring=js)


###Search the best combination of parameters for the random forest model
grid_search_1.fit(X_train, y_train)
############The best combination of parameter values is:
print(grid_search_1.best_params_)

best_grid_1 = grid_search_1.best_estimator_
###Set the the best valyes founded for the random forest model
RF_O=RandomForestClassifier(n_estimators=20, 
                                       criterion='gini', 
                                       bootstrap=True,
                                       max_depth=8, 
                                       max_features='sqrt',
                                       min_samples_leaf= 1,
                                       min_samples_split= 2,
                                       n_jobs=-1)
RF_O.fit(X_train,y_train)

plt.figure(figsize=(9,8))
plt.barh(X.columns[RF_O.feature_importances_.argsort()],RF_O.feature_importances_[RF_O.feature_importances_.argsort()])

cm=confusion_matrix(y_test,RF_O.predict(X_test))
ac_score=accuracy_score(y_test,RF_O.predict(X_test))
precision_macro=precision_score(y_true=y_test,y_pred=pd.Series(RF_O.predict(X_test)),average='macro',labels=None)
precision_micro=precision_score(y_true=y_test,y_pred=pd.Series(RF_O.predict(X_test)),average='micro',labels=None)
f1_macro=f1_score(y_test,RF_O.predict(X_test),average='macro',labels=None)
f1_micro=f1_score(y_test,RF_O.predict(X_test),average='micro',labels=None)
plot_confusion_matrix(cm, target_names=['Approbed','Failed'])
print('The confusion matrix with the test data is:')
print(cm)
print('The accuracy of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The macro precision of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The micro precision of the model with the test data is:' + ' ' +str(round(precision_micro,2)))
print('The macro f1 score of the model with the test data is:' + ' ' +str(round(f1_macro,2)))
print('The micro f1 score of the model with the test data is:' + ' ' +str(round(f1_micro,2)))

###############################################################################
#################SUPORT VECTOR MACHINE MODEL########################
param_grid_2 = {
    'kernel':['linear','poly','rbf','sigmoid'],
    'C' : [0.01, 0.1, 1, 10],
    'gamma' : [ 0.01, 0.1, 1,2,10],
    'degree': [1,2,3,4,5,6,8]}
    
svc_f=SVC(class_weight='balanced')

grid_search_2 = GridSearchCV(estimator = svc_f, param_grid = param_grid_2, 
                          cv = 10, n_jobs = -1, verbose = 2, scoring=js)
grid_search_2.fit(X_train, y_train)
print(grid_search_2.best_params_)
best_grid = grid_search_2.best_estimator_
SVC_O=SVC(C=1, class_weight='balanced',cache_size=200, coef0=0.0,
  decision_function_shape='ovr',  gamma=1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC_O.fit(X_train,y_train)

cm=confusion_matrix(y_test,SVC_O.predict(X_test))
ac_score=accuracy_score(y_test,SVC_O.predict(X_test))
precision_macro=precision_score(y_true=y_test,y_pred=pd.Series(SVC_O.predict(X_test)),average='macro',labels=None)
precision_micro=precision_score(y_true=y_test,y_pred=pd.Series(SVC_O.predict(X_test)),average='micro',labels=None)
f1_macro=f1_score(y_test,SVC_O.predict(X_test),average='macro',labels=None)
f1_micro=f1_score(y_test,SVC_O.predict(X_test),average='micro',labels=None)

plot_confusion_matrix(cm, target_names=['Approbed','Failed'])
print('The confusion matrix with the test data is:')
print(cm)
print('The accuracy of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The macro precision of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The micro precision of the model with the test data is:' + ' ' +str(round(precision_micro,2)))
print('The macro f1 score of the model with the test data is:' + ' ' +str(round(f1_macro,2)))
print('The micro f1 score of the model with the test data is:' + ' ' +str(round(f1_micro,2)))

##########Using the SMOTE algorith for resampling subrepresented samplings in
#the training sample##################
########Getting the resampled train features and train labels
sm = SMOTE(random_state=2,sampling_strategy=1)
X_smote, y_smote = sm.fit_sample(X_train, y_train)
###Search the best combination of parameters for the random forest model
random_f=RandomForestClassifier()
grid_search_1.fit(X_smote, y_smote)
grid_search_1.best_params_

###Set the the best valyes founded for the random forest model
RF_O=RandomForestClassifier(n_estimators=100, 
                                       criterion='gini', 
                                       bootstrap=True,
                                       max_depth=8, 
                                       max_features='auto',
                                       min_samples_leaf= 1,
                                       min_samples_split= 2,
                                       n_jobs=-1)
RF_O.fit(X_smote,y_smote)

plt.figure(figsize=(9,8))
plt.barh(X.columns[RF_O.feature_importances_.argsort()],RF_O.feature_importances_[RF_O.feature_importances_.argsort()])

cm=confusion_matrix(y_test,RF_O.predict(X_test))
ac_score=accuracy_score(y_test,RF_O.predict(X_test))
precision_macro=precision_score(y_true=y_test,y_pred=pd.Series(RF_O.predict(X_test)),average='macro',labels=None)
precision_micro=precision_score(y_true=y_test,y_pred=pd.Series(RF_O.predict(X_test)),average='micro',labels=None)
f1_macro=f1_score(y_test,RF_O.predict(X_test),average='macro',labels=None)
f1_micro=f1_score(y_test,RF_O.predict(X_test),average='micro',labels=None)
plot_confusion_matrix(cm, target_names=['Approbed','Failed'])
print('The confusion matrix with the test data is:')
print(cm)
print('The accuracy of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The macro precision of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The micro precision of the model with the test data is:' + ' ' +str(round(precision_micro,2)))
print('The macro f1 score of the model with the test data is:' + ' ' +str(round(f1_macro,2)))
print('The micro f1 score of the model with the test data is:' + ' ' +str(round(f1_micro,2)))

############Smote technique applied to SVM model############
svc_f=SVC()
grid_search_2.fit(X_smote, y_smote)
print(grid_search_2.best_params_)

best_grid = grid_search_2.best_estimator_

SVC_O=SVC(C=10,cache_size=200, coef0=0.0,
  decision_function_shape='ovr',  gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
SVC_O.fit(X_smote,y_smote)

cm=confusion_matrix(y_test,SVC_O.predict(X_test))
ac_score=accuracy_score(y_test,SVC_O.predict(X_test))
precision_macro=precision_score(y_true=y_test,y_pred=pd.Series(SVC_O.predict(X_test)),average='macro',labels=None)
precision_micro=precision_score(y_true=y_test,y_pred=pd.Series(SVC_O.predict(X_test)),average='micro',labels=None)
f1_macro=f1_score(y_test,SVC_O.predict(X_test),average='macro',labels=None)
f1_micro=f1_score(y_test,SVC_O.predict(X_test),average='micro',labels=None)
plot_confusion_matrix(cm, target_names=['Approbed','Failed'])
print('The confusion matrix with the test data is:')
print(cm)
print('The accuracy of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The macro precision of the model with the test data is:' + ' ' +str(round(precision_macro,2)))
print('The micro precision of the model with the test data is:' + ' ' +str(round(precision_micro,2)))
print('The macro f1 score of the model with the test data is:' + ' ' +str(round(f1_macro,2)))
print('The micro f1 score of the model with the test data is:' + ' ' +str(round(f1_micro,2)))

#Results:
#    The task selected for this project has particular features: first the original task
#    was a regression problem due to the target variable was an continuous one. Based on
#    the information available on internet this variable was discretized taking into account
#    the minimum value for approving a course. Since that the course failing is a very rare
#    event the classes were unbalanced and the models tend to overestimate the importance of
#    the 'Approving' event which causes problems in the model fitting. Two models were
#    essayed for fitting the target label: random forest and support vector machines.
#    For tunning the parametters the model an intensive searching was implemented 
#    and using the Jaccard index (recomended for unbalanced classes) the best combination
#    of parammeters was selected for fitting the models. As we expected, the models fitted 
#    the most common event very well while the rare event was underepresented in the model.
#    In a second step, a resampling technique was used for improving the results. This technique
#    is used for set the training sets, correcting the underepresentation of the target event.
#    A light improving was observed in the random forest model while the same results were observed
#    in the support vector machine model. The most important features in the model were the school origin
#    previous faiure courses, mother education, age, weekend alcohol comsumption, time devoted to study
#    father's education, time devoted to travel, free time, among others. Additional variables were
#    ommited from model in the exploratory previous step.