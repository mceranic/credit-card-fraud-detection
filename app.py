import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef

# Definition of functions

def countplot_data(data, feature):
    plt.figure(figsize=(10,10))
    sns.countplot(x=feature, data=data)
    plt.show()

def pairplot_data_grid(data, feature1, feature2, target):
    sns.FacetGrid(data, hue=target).map(plt.scatter, feature1, feature2).add_legend()
    plt.show()

def grid_evaluation(grid_clf):
    print("Best score: ", grid_clf.best_score_)
    print("Best Parameter: ", grid_clf.best_params_)

def evaluation(y_test, grid_clf, X_test):
    y_pred = grid_clf.predict(X_test)
    print("Classification report: ", classification_report(y_test, y_pred))
    print("AUC-ROC: ", roc_auc_score(y_test, y_pred))
    print("F1-Score: ", f1_score(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))

# Main code
# Import data into dataframe
creditcard_df = pd.read_csv('creditcard.csv')

# Overview of the dataset
# print(creditcard_df)
print(creditcard_df.describe())
# print(creditcard_df.columns)
print(creditcard_df.isna().sum())

# Plotting 
# Plot count values for Class variable
countplot_data(creditcard_df, creditcard_df.Class)
# Plot Amount/Time values for Class = 1/0
pairplot_data_grid(creditcard_df, "Time", "Amount", "Class")

# check how high the amount is for fraud transactions
amount_more = 0
amount_less = 0
for i in range(creditcard_df.shape[0]):
    if(creditcard_df.iloc[i]["Amount"] < 2500):
        amount_less += 1
    else:
        amount_more += 1

print("Amount > 2500: ", amount_more)
print("Amount < 2500: ", amount_less)

percentage_less = (amount_less/creditcard_df.shape[0])*100
print("Percentage of Amount < 2500: ", percentage_less)

fraud = 0
non_fraud = 1
for i in range(creditcard_df.shape[0]):
    if(creditcard_df.iloc[i]["Amount"] < 2500):
        if(creditcard_df.iloc[i]["Class"] == 0):
            non_fraud += 1
        else:
            fraud += 1
        
#print("Count of fraud records with amount < 2500: ", fraud)
#print("Count of non-fraud records with amount < 2500: ", non_fraud)
#print(creditcard_df.value_counts()) 

# Relationship between Time and Transactions
sns.FacetGrid(creditcard_df, hue="Class").map(sns.histplot, "Time").add_legend()
#plt.show()

#plt.figure(figsize=(20,20))
df_corr = creditcard_df.corr()
sns.heatmap(df_corr)

# Train and test dataset in ratio 70:30
#Features
X = creditcard_df.drop(labels='Class', axis=1)
#Target variable
y = creditcard_df.loc[:,'Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# oversampling the minority class
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
mutual_infos = pd.Series(data=mutual_info_classif(X_res, y_res, discrete_features=False, random_state=1), index=X_train.columns)
print(mutual_infos.sort_values(ascending=False))
#sns.countplot(y_res)

#Checking the score for different algorithms

param_grid_sgd = [{
    'model__loss': ['log'],
    'model__penalty': ['l1', 'l2'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20)
}, {
    'model__loss': ['hinge'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20),
    'model__class_weight': [None, 'balanced']
}]

pipeline_sgd = Pipeline([
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])

MCC_scorer = make_scorer(matthews_corrcoef)
grid_sgd = GridSearchCV(estimator=pipeline_sgd, param_grid=param_grid_sgd, scoring=MCC_scorer, n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
grid_sgd.fit(X_res, y_res)

print(grid_evaluation(grid_sgd))
print(evaluation(y_test, grid_sgd, X_test))
