# Week 1 — Activity 1: Iris Dataset Analysis

## Understanding

This activity analyses the classic **Iris Plants Database** from the UCI Machine Learning Repository (dataset id `53`). Each row describes one iris flower using four numeric measurements, and the target is the species name.

The data is loaded in `activity1.py` with `ucimlrepo.fetch_ucirepo`, which returns:

- `X` — feature columns (pandas DataFrame)
- `y` — class / target column (pandas DataFrame)

These are combined into one DataFrame so duplicate rows can be checked with pandas.

| Column | Meaning | Role |
|--------|---------|------|
| sepal length | Length of the sepal (cm) | Feature |
| sepal width | Width of the sepal (cm) | Feature |
| petal length | Length of the petal (cm) | Feature |
| petal width | Width of the petal (cm) | Feature |
| class | Species name | Target / label |

There are **150 instances** (**50 per class**) and no missing values.

---

## Findings

### 1. How many features and classes are available?

| Item | Count | Details |
|------|------:|---------|
| **Features** | **4** | `sepal length`, `sepal width`, `petal length`, `petal width` |
| **Classes** | **3** | `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` |

Class distribution is balanced: 50 samples in each class.

### 2. Are there any duplicate records in the dataset?

**Yes.** Exact full-row duplicates (same feature values and same class) exist:

| Duplicate row values | Index positions | Times |
|----------------------|-----------------|------:|
| `4.9, 3.1, 1.5, 0.1, Iris-setosa` | 9, 34, 37 | 3 |
| `5.8, 2.7, 5.1, 1.9, Iris-virginica` | 101, 142 | 2 |

Summary:

- **3** duplicate rows beyond the first occurrence of each unique row
- After removing exact duplicates, **147** unique records remain

---

## Steps to Follow

1. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run the analysis**
   ```bash
   python3 activity1.py
   ```

3. **What the script does**
   - Fetches the Iris dataset from UCI (`id=53`)
   - Prints metadata and variable information
   - Reports number of features and classes
   - Combines features + class into one DataFrame
   - Finds and prints duplicate records with `df.duplicated()`
   - Saves a petal length vs petal width scatter plot to `output.png`

4. **Review findings**
   - Confirm: 4 features, 3 classes, duplicates present
   - Decide whether to drop duplicates before any later modeling (`df.drop_duplicates()`)
   - Open `output.png` to see class separation in petal features

5. **(Optional next steps)**
   - Split into train/test sets
   - Train a simple classifier (e.g. k-NN, decision tree)
   - Check which classes are harder to separate (`versicolor` vs `virginica`)

---

## Project files

| File | Purpose |
|------|---------|
| `activity1.py` | Fetch Iris from UCI, report features/classes, find duplicates, save plot |
| `requirements.txt` | Python package dependencies |
| `output.png` | Scatter plot of petal length vs petal width by class |
| `README.md` | Understanding, findings, and steps to follow |
| `iris.zip` | Local copy of the Iris dataset (optional / backup) |
