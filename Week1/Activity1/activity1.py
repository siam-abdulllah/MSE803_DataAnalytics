from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

# combine features + class into one dataframe
df = pd.concat([X, y], axis=1)

print("Features:", X.shape[1], list(X.columns))
print("Classes:", y.nunique().iloc[0], list(y.iloc[:, 0].unique()))
print("Class counts:\n", y.value_counts())

duplicates = df[df.duplicated(keep=False)]
print("\nDuplicates found:", df.duplicated().sum())
print(duplicates)
