# ==========================================================
# 🏦 HDFC Loan Approval Prediction - FINAL COMPLETE CODE
# ==========================================================

# -------------------------------
# 1️⃣ Install Liabraries
# -------------------------------
!pip install -q kaggle scikit-learn seaborn

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

dataset_name = "altruistdelhite04/loan-prediction-problem-dataset"
!kaggle datasets download -d $dataset_name
!unzip -o *.zip

print("✅ Dataset Downloaded & Extracted!")

# ==========================================================
# 4️⃣ Load Training Dataset Automatically
# ==========================================================

train_file = [f for f in glob.glob("*.csv") if "train" in f.lower()][0]
print("📂 Loading file:", train_file)

df = pd.read_csv(train_file)
print("✅ Dataset Loaded!")
print("Shape:", df.shape)

# ==========================================================
# 5️⃣ Data Preprocessing
# ==========================================================

# Drop Loan_ID if present
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Handle Missing Values
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Feature Engineering
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']

# Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ==========================================================
# 6️⃣ Visualization
# ==========================================================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Credit History vs Loan Status")
plt.show()

# ==========================================================
# 7️⃣ Define Features & Target
# ==========================================================

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# 8️⃣ Model Training
# ==========================================================

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==========================================================
# 9️⃣ Evaluation Function
# ==========================================================

def evaluate(y_test, y_pred, name):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

evaluate(y_test, y_pred_lr, "Logistic Regression")
evaluate(y_test, y_pred_dt, "Decision Tree")
evaluate(y_test, y_pred_rf, "Random Forest")

# ====================================