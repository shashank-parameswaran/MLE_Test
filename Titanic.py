import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

#os.chdir("C:\\Data Science Projects\\Kaggle\\titanic")

train = pd.read_csv("./data/train.csv")
test_oos = pd.read_csv("./data/test.csv")
print(test_oos.shape)

combine = [train, test_oos]

for dataset in combine:
    dataset['Member'] = dataset['SibSp'] + dataset['Parch']
    #dataset['Alone'] = np.where(dataset['Member'] > 0, 0, 1)
    
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') 
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)
    
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} )
    
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    dataset['Age'] = dataset['Age'].fillna(0)
    dataset['Age'][(dataset['Age'] == 0) & (dataset['Title'] == 1)] = 35
    dataset['Age'][(dataset['Age'] == 0) & (dataset['Title'] == 2)] = 23
    dataset['Age'][(dataset['Age'] == 0) & (dataset['Title'] == 3)] = 35
    dataset['Age'][(dataset['Age'] == 0) & (dataset['Title'] == 4)] = 15
    dataset['Age'][(dataset['Age'] == 0) & (dataset['Title'] == 5)] = 40
    
    dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
    
    
    ageb = list(set(dataset['AgeBand']))
    dataset['AgeBandno'] = 0
    for i,item in zip(range(1,5), ageb):
        dataset['AgeBandno'][dataset['Age'].between(item.left, item.right)] = i
    
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean()) 
    dataset['FareBand'] = pd.cut(dataset['Fare'], 5)
    fareb = list(set(dataset['FareBand']))
    dataset['FareBandno'] = 0
    for i,item in zip(range(1,5), fareb):
        dataset['FareBandno'][dataset['Fare'].between(item.left, item.right)] = i
    #dataset = dataset.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1)
    
train_df = combine[0]
train_df = train_df.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'AgeBand', 'FareBand', 'Age', 'Fare'], axis = 1)
oos_df = combine[1]
oos_df2 = oos_df.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'AgeBand', 'FareBand', 'Age', 'Fare'], axis = 1)

x_train, x_test = train_test_split(train_df, test_size = 0.056, random_state = 42)
x_train_sub = x_train[['Pclass', 'Sex', 'Embarked', 'Member', 'Title', 'AgeBandno', 'FareBandno']]
y_train_sub = x_train['Survived']
x_test_sub = x_test[['Pclass', 'Sex', 'Embarked', 'Member', 'Title', 'AgeBandno', 'FareBandno']]
y_test_sub = x_test['Survived']

rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', min_samples_split  = 2)
rfc.fit(x_train_sub, y_train_sub)
predicted = rfc.predict(x_test_sub)
accuracy = accuracy_score(y_test_sub, predicted)
joblib.dump(rfc, "./model/rf.joblib")