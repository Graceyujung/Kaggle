 # Titanic - Machine Learning from Disaster

A predictive modeling project using the Titanic dataset to answer the question: **"What sorts of people were more likely to survive?"**

## 🚢 Overview

This project uses machine learning algorithms to explore passenger data to predict survival outcomes. It was built for the classic Kaggle competition [Titanic: Machine Learning from Disaster]([https://www.kaggle.com/c/titanic](https://www.kaggle.com/competitions/titanic/data)).

## 📊 Dataset

- **Train dataset**: 891 records with labels
- **Test dataset**: 418 records without labels
- Provided by Kaggle

## 🔧 Features Used

- `Pclass` (Ticket class)
- `Sex`
- `Age` (with imputation)
- `Fare`, `Embarked` (filled missing values)
- Engineered features: `FamilySize`, `IsAlone`, etc.

## 🛠️ Models Compared

- SVM
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

Used GridSearchCV, RandomizedSearchCV, cross-validatiom for hyperparameter tuning  and assess model performance.

## 🧪 Preprocessing

- Missing value imputation (`Age`, `Embarked`, `Fare`)
- Feature encoding (Label Encoding + One-Hot Encoding)
- Outlier detection for scaling
- Feature scaling (RobustScaler)
  
## ✅ Best Model

- **Model**: [SVM_rbf]
- **Accuracy on validation set**: [0.85]
- **Kaggle submission score**: [0.78229]

## 📁 File Structure

```bash
titanic/
├── train.csv
├── test.csv
├── main.ipynb
├── train.ipynb
├── test.ipynb
├── submission.csv
└── README.md

