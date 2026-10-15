
import pandas as pd

# Load the dataset downloaded from Azure
df = pd.read_csv("/tmp/ai4i2020.csv")

print("Dataset dimensions:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nMachine failure distribution:")
print(df["Machine failure"].value_counts())
