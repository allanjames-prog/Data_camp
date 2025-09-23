---

# 📘 Python & Data Science Notes

---

## 1. Python Basics (for Data Science)

* **Definition:** Python is the most popular language for data science due to its readability, flexibility, and rich ecosystem.
* **Key Features:**

  * Dynamically typed, interpreted, supports multiple paradigms.
  * Extensive libraries for data analysis, visualization, and machine learning.

### Variables & Data Types

```python
x = 5        # int (used for counting, indexing)
pi = 3.14    # float (used for statistics, calculations)
name = "Ali" # str (used for categorical/text data)
is_active = True # bool (used for filtering, logic)
```

* Other types: `list`, `tuple`, `dict`, `set` (often used for storing and manipulating datasets).
* Type checking: `type(x)` (important for data validation).

### Operators

* Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**` (used in calculations, feature engineering).
* Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=` (used for filtering data).
* Logical: `and`, `or`, `not` (used in complex filtering and conditional logic).

### Control Flow

```python
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")
```
* Used to make decisions based on data values.

### Loops

```python
for i in range(5):
    print(i)  # Useful for iterating over rows, columns, or features

while x > 0:
    print(x)  # Useful for repeated operations until a condition is met
    x -= 1
```
* `for` loops are common for iterating over datasets.
* `while` loops are used for iterative algorithms (e.g., convergence in machine learning).

### Indentation

* Python uses **indentation** to define code blocks, which is crucial for readable data science scripts.

---

## 2. Python Lists (for Data Science)

* **Definition:** Ordered, mutable collections. Used to store sequences of data (e.g., feature lists, results).
* Can hold **heterogeneous data** (e.g., `[1, "hello", 3.5]`).

### Basic Operations

```python
my_list = [1, 2, 3]
print(my_list[0])       # Access
my_list.append(4)       # Add (e.g., new data point)
my_list.remove(2)       # Remove (e.g., cleaning data)
my_list[1:3]            # Slicing (e.g., selecting subsets)
```

### Useful Methods

* `len(my_list)` → length (size of dataset)
* `my_list.sort()` → sort in-place (ranking, ordering)
* `sorted(my_list)` → returns new sorted list
* `my_list.reverse()` → reverse order
* `my_list.index(3)` → get index of value
* `my_list.count(2)` → count occurrences (frequency analysis)

### Nested Lists

```python
matrix = [[1,2], [3,4]]
print(matrix[0][1])  # Access 2 (useful for representing tabular data)
```

---

## 3. Functions and Packages (for Data Science)

* **Functions:** Reusable blocks of code for data processing, cleaning, analysis.

```python
def greet(name="User"):
    return f"Hello, {name}"
```

* **Arguments:** Allow flexibility in data processing functions.
  * Positional, keyword, default, `*args`, `**kwargs` (handle variable data inputs).

* **Lambda functions:** Useful for quick data transformations.

```python
square = lambda x: x**2
```

* **Packages & Modules:** Extend Python for data science.
  * `math`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` are essential.
  * Importing modules enables advanced data operations.

---

## 4. NumPy (for Data Science)

* **Purpose:** Fast numerical operations on large datasets (arrays, matrices).
* **Core Object:** `ndarray` — multidimensional array with homogeneous elements.
* Used for mathematical, statistical, and linear algebra operations.

### Creating Arrays

```python
arr = np.array([1, 2, 3])
zeros = np.zeros((2,3))
ones = np.ones(5)
rand = np.random.rand(3,2)
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 5)
```

### Operations

* Element-wise: `arr * 2`, `arr + 5` (feature scaling, transformation)
* Aggregation: `np.sum(arr)`, `np.mean(arr)`, `np.median(arr)`, `np.std(arr)` (statistics)
* Matrix multiplication: `np.dot(a, b)` or `a @ b` (machine learning, linear algebra)
* Transpose: `arr.T`
* Boolean indexing: `arr[arr > 5]` (filtering data)

### Advanced

* Reshaping: `arr.reshape((3,2))` (preparing data for models)
* Stacking: `np.vstack`, `np.hstack` (combining datasets)
* Broadcasting: operations on arrays of different shapes (efficient computation)

### Handling Missing Data

* Use `np.nan` to represent missing values.
* Check for missing values: `np.isnan(arr)`
* Ignore missing values in calculations: `np.nanmean(arr)`, `np.nanstd(arr)`

### Random Sampling

* Generate random samples: `np.random.choice(arr, size=5)` (bootstrapping, sampling)
* Random numbers: `np.random.randn(100)` (simulation, initialization)

---

## 5. Matplotlib (for Data Science)

* **Purpose:** Visualization library for Python. Essential for data exploration and communication.
* Commonly used module: `matplotlib.pyplot`.

### Basic Example

```python
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [10,20,25,30]

plt.plot(x, y, 'o--r', label='Line')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Plot")
plt.legend()
plt.show()
```

### Common Plot Types

* Line: `plt.plot(x,y)` (trend analysis)
* Scatter: `plt.scatter(x,y)` (correlation, outlier detection)
* Bar: `plt.bar(x,y)` (category comparison)
* Histogram: `plt.hist(data, bins=10)` (distribution analysis)
* Pie: `plt.pie(sizes, labels=labels)` (proportion visualization)
* Box: `plt.boxplot(data)` (spread, outliers)

### Customization

* Colors, markers, subplots, figure size, grid, save figure.
* Save plots for reports: `plt.savefig('plot.png')`

### Integration with pandas

* Plot directly from DataFrames: `df.plot()` (quick EDA)

### Applications in Data Science

* EDA: visualize distributions, trends, relationships.
* Model evaluation: plot predictions vs. actual, confusion matrices, ROC curves.
* Publication-quality figures.

---

## 6. Python Dictionaries (for Data Science)

* **Definition:** Unordered, mutable collections of key-value pairs.
* Used for mapping categorical data, storing configurations, counting occurrences.

### Example

```python
person = {"name": "James", "age": 25, "is_student": True}
print(person["name"])
person["city"] = "Kampala"
```

### Operations

* Access: `person.get("grade", "N/A")`
* Delete: `del person["age"]` or `person.pop("age")`
* Keys/Values: `person.keys()`, `person.values()`
* Iteration:

  ```python
  for key, value in person.items():
      print(key, value)
  ```

### Dictionary Comprehensions

* Create new dictionaries from iterables:
  ```python
  squares = {x: x**2 for x in range(5)}
  ```

### Applications

* Data mapping (e.g., `{ "M": 1, "F": 0 }` for encoding).
* Counting with `collections.Counter` (frequency analysis).
* JSON-like data structures (data exchange).
* Configurations and settings storage.

---

## 7. Pandas (for Data Science)

* **Pandas** is the most widely used library for data manipulation and analysis.
* Provides:
  - **Series:** 1D labeled array (single column).
  - **DataFrame:** 2D labeled table (rows and columns).

### Creating Data Structures

```python
import pandas as pd

# Series
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# DataFrame from dictionary
df = pd.DataFrame({
    'Name': ['Ali', 'James', 'Sara'],
    'Age': [25, 30, 22]
})
```

### Reading and Writing Data

* Read CSV: `pd.read_csv('data.csv')`
* Read Excel: `pd.read_excel('data.xlsx')`
* Write CSV: `df.to_csv('output.csv', index=False)`

### DataFrame Operations

* View data: `df.head()`, `df.tail()`
* Info: `df.info()`, `df.describe()` (summary statistics)
* Select column: `df['Name']`
* Select row by label: `df.loc[0]`
* Select row by index: `df.iloc[0]`
* Filter rows: `df[df['Age'] > 25]`
* Add column: `df['Score'] = [90, 85, 88]`
* Drop column: `df.drop('Score', axis=1)`

### Handling Missing Data

* Detect missing: `df.isnull()`
* Drop missing: `df.dropna()`
* Fill missing: `df.fillna(0)`

### Grouping and Aggregation

* Group by column: `df.groupby('Name').mean()`
* Aggregate: `df.agg({'Age': ['mean', 'max']})`

### Merging and Joining

* Merge DataFrames: `pd.merge(df1, df2, on='key')`
* Concatenate: `pd.concat([df1, df2], axis=0)`

### Pivot Tables

* Create pivot tables: `df.pivot_table(values='Sales', index='Date', columns='Region')`

### Time Series

* Convert to datetime: `pd.to_datetime(df['date_column'])`
* Set index: `df.set_index('date_column')`
* Resample: `df.resample('M').sum()`

### Applications

* Data cleaning and preparation.
* Exploratory data analysis (EDA).
* Feature engineering for machine learning.
* Data visualization integration.

---

## Comparison Operators in Data Science

* Used to filter, compare, and select data in analysis.

| Operator | Description           | Example         | Result      |
|----------|----------------------|-----------------|-------------|
| `==`     | Equal to             | `df['Age'] == 25`        | Boolean Series |
| `!=`     | Not equal to         | `df['Name'] != 'Ali'`    | Boolean Series |
| `>`      | Greater than         | `df['Score'] > 80`       | Boolean Series |
| `<`      | Less than            | `df['Age'] < 30`         | Boolean Series |
| `>=`     | Greater than or equal| `df['Age'] >= 25`        | Boolean Series |
| `<=`     | Less than or equal   | `df['Score'] <= 90`      | Boolean Series |

### Usage Example

```python
filtered = df[df['Age'] > 25]  # Select rows where Age > 25
```

* Essential for data filtering and conditional selection.

---

## Boolean Operators in Data Science

* Combine multiple conditions for filtering and selection.

| Operator | Description         | Example             | Result      |
|----------|--------------------|---------------------|-------------|
| `and`    | Logical AND        | `(df['Age'] > 20) & (df['Score'] > 85)` | Boolean Series |
| `or`     | Logical OR         | `(df['Age'] < 25) | (df['Score'] > 90)` | Boolean Series |
| `not`    | Logical NOT        | `~(df['Name'] == 'Ali')` | Boolean Series |

### Usage Example

```python
filtered = df[(df['Age'] > 20) & (df['Score'] > 85)]
```

* Used for complex data queries and cleaning.

---

## Conditional Statements in Data Science

* Used for decision making, data transformation, and control flow.

### Syntax

```python
if condition:
    # code block if condition is True
elif another_condition:
    # code block if another_condition is True
else:
    # code block if none of the above are True
```

### Example

```python
if df['Age'].mean() > 25:
    print("Average age is above 25")
else:
    print("Average age is 25 or below")
```

* Often used in data cleaning, feature engineering, and custom analysis.

---

## While Loops in Data Science

* **While loops** repeatedly execute a block of code as long as a condition is `True`.
* Useful for iterative processes, such as cleaning data, training models until convergence, or searching for a value.

### Syntax

```python
while condition:
    # code block
```

### Example: Counting Down

```python
x = 5
while x > 0:
    print(x)
    x -= 1
# Output: 5 4 3 2 1
```

### Example: Data Cleaning

```python
# Remove all negative values from a list
data = [5, -2, 7, -1, 3]
while min(data) < 0:
    data.remove(min(data))
print(data)  # Output: [5, 7, 3]
```

### Example: Iterative Algorithm (e.g., convergence)

```python
error = 1.0
while error > 0.01:
    # update model, recalculate error
    error = error / 2
    print("Current error:", error)
```

### Notes

* Be careful to update the condition inside the loop to avoid infinite loops.
* While loops are less common than `for` loops in data science, but are essential for tasks that require repeated checking or updating until a condition is met.

