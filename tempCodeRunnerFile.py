import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load CSV
df = pd.read_csv('medical_data.csv')

# Display first 5 rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# Convert data types if needed
df['Age'] = df['Age'].astype(int)

print("Total Patients:", len(df))
print("Average Age:", df['Age'].mean())
print("Most Common Diagnosis:", df['Diagnosis'].mode()[0])

# Count of each diagnosis
print(df['Diagnosis'].value_counts())

# Average Glucose by Diagnosis
print(df.groupby('Diagnosis')['GlucoseLevel'].mean())


# Plot diagnosis counts
sns.countplot(x='Diagnosis', data=df)
plt.title("Diagnosis Count")
plt.show()

# Boxplot of glucose levels by diagnosis
sns.boxplot(x='Diagnosis', y='GlucoseLevel', data=df)
plt.title("Glucose Level by Diagnosis")
plt.show()

# Save cleaned and analyzed data to Excel
df.to_excel("cleaned_medical_data.xlsx", index=False)
