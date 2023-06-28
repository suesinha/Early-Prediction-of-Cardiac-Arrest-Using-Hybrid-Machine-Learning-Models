import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#loading Data
from google.colab import files
uploaded = files.upload()
data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
#reading data information
data.info()
data.head()
print(f'data shape : {data.shape}')#data Statistics
data["HeartDiseaseorAttack"].value_counts()
plt.title("Relation of BMI with cardiac arrest")
plt.hist(data.BMI)
#Sample EDA
sns.countplot(x='Smoker',hue='HeartDiseaseorAttack',data=data)
sns.countplot(x='Education',hue='HeartDiseaseorAttack',data=data)
sns.countplot(x='HighBP',hue='HeartDiseaseorAttack',data=data)
sns.heatmap(data)
#data statistics
cor_matrix = data.corr().abs()
cor_matrix
# correlation matrix
correlation_matrix = data.corr()
k = 22 # number of variables for heatmap
cols = correlation_matrix.nlargest(k,'HeartDiseaseorAttack')['HeartDiseaseorAttack'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(10,10))  # Sample figsize in inches
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.01f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,ax=ax)
plt.title("Correlation Matrix")
plt.show()
#remove duplicate correlation diagonal
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
upper_tri
#drop both highly correlated columns
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7) or any(upper_tri[column] < 0.01)]
to_drop
#specify features and target columns
target = data['HeartDiseaseorAttack']
features = data.drop(to_drop, axis=1)
features = features.drop('HeartDiseaseorAttack',axis=1)
features.info()
plt.plot(features)
sns.heatmap(features)
#Data Scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
#data splitting
X_train, X_test, y_train, y_test = train_test_split(scaled_features,target,stratify=target, test_size=0.25)
y_train.value_counts()
from sklearn.naive_bayes import GaussianNB
Gnb = GaussianNB()
Gnb.fit(X_train, y_train)
y_pred = Gnb.predict(X_test)
print("Naive Bayes Classifier Accuracy: ",accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Classifier Accuracy: ",accuracy_score(y_test, y_pred))
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
print("XGBClassifier Accuracy: ",accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,classification_report
models={
    'DT':DecisionTreeClassifier(),
    'SVM':SVC()
}
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
svm = SVC()

for name,model in  models.items():
    print(f'using {name}: ')
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f'Training Accuracy :{accuracy_score(y_train,model.predict(X_train))}')
    print(f'Testing Accuracy :{accuracy_score(y_test,y_pred)}')
    print(f'Confusion matrix:\n {confusion_matrix(y_test,y_pred)}')
    print(f'Recall: {recall_score(y_test,y_pred)}')
    print(f'precision: {precision_score(y_test,y_pred)}')
    print(f'F1-score: {f1_score(y_test,y_pred)}')
    print(classification_report(y_test,y_pred))
    print('-'*33)
from sklearn import metrics
plt.figure(0).clf()
y_pred = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

y_pred = Gnb.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Random Forest, AUC="+str(auc))

y_pred = xg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Hybrid Model, AUC="+str(auc))

y_pred = dt.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Naive Bayes, AUC="+str(auc)) 



plt.legend()
