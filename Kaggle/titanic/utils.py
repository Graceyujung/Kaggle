import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from statistics import mean

def preprocess_data(data, is_train=True):
    
    data = data.copy()
    
    passenger_ids = data['PassengerId'] if not is_train else None
    
    data['TicketCode'] = data['Ticket'].str.extract(r'(\d+)', expand=False)
    data['TicketCode'] = data['TicketCode'].fillna('Unknown')

    # Fill missing 'Age' values based on the median of each 'Pclass'
    data['Age'] = data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

    # Fill missing 'Fare' values with the median
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
  
    # Fill missing 'Embarked' values with the mode (most frequent value)
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Create a 'FamilySize' column by adding 'SibSp' (siblings/spouses aboard) and 'Parch' (parents/children aboard) + 1 for the individual
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  # +1 for self
    # Create an 'IsAlone' column where 1 means the person is alone (FamilySize == 1), otherwise 0
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    drop_cols = ['Cabin', 'Name', 'Ticket', 'SibSp', 'Parch', 'PassengerId']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    if not is_train:
        return data, passenger_ids
    else:
        return data

def encode_feature(data):
 
    # Map 'Sex' column to numerical values (0 for male, 1 for female)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # One-hot encoding for 'Embarked'
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    data[['Embarked_Q', 'Embarked_S']] = data[['Embarked_Q', 'Embarked_S']].astype(int)

    return data

def detect_outliers_iqr(data, column):
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data, data[(data[column] < lower_bound) | (data[column] > upper_bound)]

def splitting_data(data, return_all=True):

    X = data.drop('Survived', axis=1)
    Y = data['Survived']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Size of training data: {X_train.shape[0]}")
    print(f"Size of test data: {X_test.shape[0]}")

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_to_scale = ['Age', 'Fare', 'FamilySize']
    rs = RobustScaler()

    X_train_scaled[cols_to_scale] = rs.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = rs.transform(X_test[cols_to_scale])
    
    joblib.dump(rs, 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test  

def evaluate_model(model, X_train_scaled, X_test_scaled, Y_train, Y_test):

    model.fit(X_train_scaled, Y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Accuracy
    print(f"Train Accuracy: {accuracy_score(Y_train, y_train_pred):.2f}")
    print(f"Test Accuracy: {accuracy_score(Y_test, y_test_pred):.2f}")

    # Confusion Matrix (Train)
    print(f"\nConfusion Matrix (Train):\n {confusion_matrix(Y_train, y_train_pred)}")

    # Confusion Matrix (Test)
    print(f"\nConfusion Matrix (Test):\n {confusion_matrix(Y_test, y_test_pred)}")

    target_names = ['Not Survived', 'Survived']
    
    # Classification Report (Train)
    print("\nClassification Report (Train):")
    print(classification_report(Y_train, y_train_pred, target_names=target_names))

    # Classification Report (Test)
    print("\nClassification Report (Test):")
    print(classification_report(Y_test, y_test_pred, target_names=target_names))

    return model