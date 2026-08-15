"""Clean the sample dataset, then compute basic statistics with NumPy."""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
df = pd.read_csv("Sample_dataset.csv")

# Merge Bob's two incomplete rows into one
df = df.groupby("ID", dropna=False, as_index=False).first()

# Fix known messy values
df["ID"] = df["ID"].fillna(3).astype(int)
df["Name"] = df["Name"].fillna("Unknown")
df["Age"] = df["Age"].replace("thirty-eight", 38)
df["Net worth"] = df["Net worth"].replace("30,000", "30000")
df["Salary"] = df["Salary"].replace("sixty five thousand", 65000)
df["Country"] = df["Country"].replace({"AU": "AUS"}).fillna("Unknown")

# Convert numbers and dates
for col in ["Age", "Net worth", "Salary"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Join Date"] = pd.to_datetime(df["Join Date"], dayfirst=True, errors="coerce")
# 2019-13-01 is invalid; treat it as 13 Jan 2019
df.loc[df["Name"] == "Eve", "Join Date"] = pd.Timestamp("2019-01-13")

# Drop rows with no numeric data (Heidi)
df = df.dropna(subset=["Age", "Net worth", "Salary"], how="all")
df = df.sort_values("ID").reset_index(drop=True)
df.to_csv("cleaned_dataset.csv", index=False)

lines = []

def out(text=""):
    print(text)
    lines.append(text)

out("CLEANED DATA")
out(df.to_string(index=False))

# ddof=0 divides by n; ddof=1 divides by n-1
out("\nRESULTS")
for col in ["Age", "Net worth", "Salary"]:
    x = df[col].dropna().to_numpy(dtype=float)
    out(
        f"{col}: n={len(x)}  mean={np.mean(x):,.2f}  "
        f"median={np.median(x):,.2f}  var={np.var(x):,.2f}  "
        f"std={np.std(x):,.2f}"
    )

out()
for a, b in [("Age", "Salary"), ("Age", "Net worth"), ("Salary", "Net worth")]:
    pair = df[[a, b]].dropna().to_numpy(dtype=float)
    out(
        f"{a} vs {b}: cov={np.cov(pair[:, 0], pair[:, 1], ddof=1)[0, 1]:,.2f}  "
        f"r={np.corrcoef(pair[:, 0], pair[:, 1])[0, 1]:.3f}"
    )

out("""
n            count of values
mean         typical value (sum / n)
median       middle value; less affected by outliers
variance     spread, in squared units
std          typical distance from the mean
covariance   + together, - opposite; look at the sign
correlation  same idea, scaled to -1 to +1
""")

with open("analytics_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Saved analytics_results.txt")
