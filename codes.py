import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load data
data_path = "C:\\Vs Codes\\Ml case study\\customer_data.csv"
df = pd.read_csv(data_path)

# Drop unnecessary columns
df.drop(['RowNumber', 'Surname', 'Gender', 'Geography', 'CustomerId'], axis=1, inplace=True)

# Exploratory Data Analysis (EDA)
def plot_value_counts(column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f"Distribution of {column}")
    plt.show()

# Plot distribution of 'Exited'
plot_value_counts('Exited')

# Display summary of categorical columns
def display_category_summary(dataframe, columns):
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=dataframe[col], data=dataframe)
        plt.title(f"Count plot for {col}")
        plt.show()
        print(pd.DataFrame({col: dataframe[col].value_counts(), "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}))
        print("##########################################")

# Identify categorical columns with low unique values
category_columns = [col for col in df.columns if df[col].nunique() < 10]
display_category_summary(df, category_columns)

# Compute probabilities for specific conditions
def compute_probabilities(df, feature, target, value):
    prob = round(df[(df[feature] == value) & (df[target] == 1)].shape[0] / df[df[target] == 1].shape[0] * 100, 2)
    return prob

credit_card_prob = compute_probabilities(df, "HasCrCard", "Exited", 1)
no_credit_card_prob = compute_probabilities(df, "HasCrCard", "Exited", 0)
active_member_prob = compute_probabilities(df, "IsActiveMember", "Exited", 1)
inactive_member_prob = compute_probabilities(df, "IsActiveMember", "Exited", 0)

print(f'Probability of churn with credit card: {credit_card_prob}%')
print(f'Probability of churn without credit card: {no_credit_card_prob}%')
print(f'Probability of churn for active members: {active_member_prob}%')
print(f'Probability of churn for inactive members: {inactive_member_prob}%')

# Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Normalizing features
scaler = MinMaxScaler()
features = df.iloc[:, :-1].values
target = df.iloc[:, -1].values
features_normalized = scaler.fit_transform(features)

# Resampling to handle imbalance
smote = SMOTE(sampling_strategy=1)
features_resampled, target_resampled = smote.fit_resample(features_normalized, target)
print(Counter(target_resampled))

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.33, random_state=1)
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(features_resampled, target_resampled, test_size=0.33, random_state=1)

# Model training and evaluation functions
def train_model(classifier, x_train, y_train):
    classifier.fit(x_train, y_train)
    return classifier

def evaluate_model(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"ROC AUC Score: {roc_auc:.2%}")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(y_test, y_pred))

# Logistic Regression
logistic_reg = LogisticRegression(max_iter=1000)
print("Logistic Regression (Imbalanced)")
logistic_reg = train_model(logistic_reg, X_train, y_train)
evaluate_model(logistic_reg, X_test, y_test)

print("Logistic Regression (Balanced)")
logistic_reg = train_model(logistic_reg, X_train_resampled, y_train_resampled)
evaluate_model(logistic_reg, X_test_resampled, y_test_resampled)

# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
print("Random Forest Classifier (Imbalanced)")
random_forest = train_model(random_forest, X_train, y_train)
evaluate_model(random_forest, X_test, y_test)

print("Random Forest Classifier (Balanced)")
random_forest = train_model(random_forest, X_train_resampled, y_train_resampled)
evaluate_model(random_forest, X_test_resampled, y_test_resampled)

# Gradient Boosting Classifier
gradient_boosting = GradientBoostingClassifier(n_estimators=100)
print("Gradient Boosting Classifier (Imbalanced)")
gradient_boosting = train_model(gradient_boosting, X_train, y_train)
evaluate_model(gradient_boosting, X_test, y_test)

print("Gradient Boosting Classifier (Balanced)")
gradient_boosting = train_model(gradient_boosting, X_train_resampled, y_train_resampled)
evaluate_model(gradient_boosting, X_test_resampled, y_test_resampled)