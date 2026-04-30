import pandas as pd

# Load the dataset
df = pd.read_csv('data/german_credit.csv')

# Basic info
print("=== SHAPE ===")
print(df.shape)  # rows x columns

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== LOAN STATUS COUNTS ===")
print(df['Loan_Status'].value_counts())