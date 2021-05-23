import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\AMXWAM_ TASK\TASK-33\creditcard.csv')
import itertools
import matplotlib.mlab as mlab
des=pd.DataFrame(dataset.describe())
# recovery of fraud data 
dataset_fraud=dataset[dataset['Class']==1]
# so there are 492 fruad cases found.
plt.scatter(dataset_fraud['Time'],dataset_fraud['Amount'],linewidths=0.2)
plt.title('fraud data recovery with respect to time and amount')
plt.xlabel('time(sec)')
plt.ylabel('Amount')
plt.xlim([0,175000])
plt.ylim([0,2500])
max(dataset['Amount'])
nb_fraud=dataset[dataset['Amount']>1000].shape[0]
print('There are only '+ str(nb_fraud) + ' frauds where the amount was bigger than 1000 over ' + str(dataset_fraud.shape[0]) + ' frauds')
df_1000=dataset_fraud[dataset_fraud['Amount']>1000]
frauds=len(dataset[dataset['Class']==1])
no_fraud=len(dataset[dataset['Class']==0])
print('there are frauds of'+str(frauds)+'along with no frauds of'+str(no_fraud))

len_fraud=len(dataset[dataset.Class==1])
len_nofraud=len(dataset[dataset.Class==0])
print('There are about'+' ' + str(len_fraud) + ' '+ 'frauds' +' '+ 'along with'+' '+str(len_nofraud)+'no frauds.')
# number of fruad cases are very less when with no fruad
#out of 284807 only 492 are fruads cases happened that will lead to imbalance of my dataset.
# in this case if u train with this dataset it will represent frauds as  no frauds when building a model 
# to make it clear lets see
(284315-492)/284315 # this could be our accuracy with unbalanced data which is not correct
datasetcorr=dataset.corr()
plt.figure(figsize=(10,7),facecolor='w')
sns.heatmap(datasetcorr, cmap="YlGnBu") # Displaying the Heatmap
sns.set(font_scale=2,style='white')
# checking the relation with class feature
rank = datasetcorr['Class'] # check with other feature # here my class(dependent) is highly correlated with other features
df_rank = pd.DataFrame(rank) 
df_rank = np.abs(df_rank).sort_values(by='Class',ascending=False) # Ranking the absolute values of the coefficients in descending order
df_rank.dropna(inplace=True) # Removing Missing Data (not a number)
# build our training dataset our training dataset is unbalanced we are divding into 0 and 1
df_train_all = dataset[0:150000] # We cut in two the original dataset
df_train_1 = df_train_all[df_train_all['Class'] == 1] # We seperate the data which are the frauds and the no frauds
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('In this dataset, we have ' + str(len(df_train_1)) +" frauds so we need to take a similar number of non-fraud")

dataset_sample=df_train_0.sample(300)
dataset_train = df_train_1.append(dataset_sample) # We gather the frauds with the no frauds. 
dataset_train = dataset_train.sample(frac=1) # Then we mix our dataset
 # in the above we equalled the fraud with no fraud
 
 # train and test split with our balanced data
dataset_train.drop(columns='Time',axis=1)
X=dataset_train.iloc[:,1:-1].values
y=dataset_train.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
sum(y_test==1)
sum(y_pred==1)
sum(y_test==0)
sum(y_pred==0)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix()
