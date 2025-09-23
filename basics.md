---

# 📘 Python & Data Science Notes

---

## 1. Python Basics

* **Definition:** Python is an interpreted, high-level, general-purpose programming language, known for readability and ease of use.
* **Key Features:**

  * Dynamically typed (no need to declare variable types explicitly).
  * Interpreted (executed line by line).
  * Supports multiple paradigms (procedural, object-oriented, functional).
  * Large standard library and third-party ecosystem.

### Variables & Data Types

```python
x = 5        # int
pi = 3.14    # float
name = "Ali" # str
is_active = True # bool
```

* Other types: `list`, `tuple`, `dict`, `set`.
* Type checking: `type(x)`

### Operators

* Arithmetic: `+`, `-`, `*`, `/`, `//` (floor), `%`, `**` (power).
* Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`.
* Logical: `and`, `or`, `not`.

### Control Flow

```python
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")
```

### Loops

```python
for i in range(5):
    print(i)

while x > 0:
    print(x)
    x -= 1
```

### Indentation

* Python uses **indentation (spaces or tabs)** instead of braces `{}` to define code blocks.

---

## 2. Python Lists

* **Definition:** Ordered, mutable collections.
* Can hold **heterogeneous data** (e.g., `[1, "hello", 3.5]`).

### Basic Operations

```python
my_list = [1, 2, 3]
print(my_list[0])       # Access
my_list.append(4)       # Add
my_list.remove(2)       # Remove
my_list[1:3]            # Slicing
```

### Useful Methods

* `len(my_list)` → length
* `my_list.sort()` → sort in-place
* `sorted(my_list)` → returns new sorted list
* `my_list.reverse()` → reverse order
* `my_list.index(3)` → get index of value
* `my_list.count(2)` → count occurrences

### Nested Lists

```python
matrix = [[1,2], [3,4]]
print(matrix[0][1])  # Access 2
```

---

## 3. Functions and Packages

* **Functions:** Reusable blocks of code.

```python
def greet(name="User"):
    return f"Hello, {name}"
```

* **Arguments:**

  * Positional: `add(2,3)`
  * Keyword: `greet(name="James")`
  * Default: `name="User"`
  * `*args` (variable arguments), `**kwargs` (keyword arguments).

* **Lambda functions:**

```python
square = lambda x: x**2
```

* **Packages & Modules:**

  * A **module** is a Python file with definitions (`.py`).
  * A **package** is a collection of modules in a directory with `__init__.py`.
  * Import:

    ```python
    import math
    from math import sqrt
    import numpy as np
    ```

---

## 4. NumPy

* **Purpose:** Fast numerical operations on large datasets.
* **Core Object:** `ndarray` — multidimensional array with homogeneous elements.
* Much faster than lists due to **vectorization** and memory efficiency.

### Creating Arrays

```python
arr = np.array([1, 2, 3])
zeros = np.zeros((2,3))
ones = np.ones(5)
rand = np.random.rand(3,2)
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 5)  # evenly spaced values
```

### Operations

* Element-wise: `arr * 2`, `arr + 5`
* Aggregation: `np.sum(arr)`, `np.mean(arr)`, `np.median(arr)`, `np.std(arr)`
* Matrix multiplication: `np.dot(a, b)` or `a @ b`
* Transpose: `arr.T`
* Boolean indexing: `arr[arr > 5]`

### Advanced

* Reshaping: `arr.reshape((3,2))`
* Stacking: `np.vstack`, `np.hstack`
* Broadcasting: allows operations on arrays of different shapes.

### Handling Missing Data

* Use `np.nan` to represent missing values.
* Check for missing values: `np.isnan(arr)`
* Ignore missing values in calculations: `np.nanmean(arr)`, `np.nanstd(arr)`

### Random Sampling

* Generate random samples: `np.random.choice(arr, size=5)`
* Random numbers: `np.random.randn(100)`

### Dictionary Comprehensions

* Create dictionaries using comprehensions:
  ```python
  squares = {x: x**2 for x in range(5)}
  ```

---

## 5. Matplotlib

* **Purpose:** Visualization library for Python.
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

* Line: `plt.plot(x,y)`
* Scatter: `plt.scatter(x,y)`
* Bar: `plt.bar(x,y)`
* Histogram: `plt.hist(data, bins=10)`
* Pie: `plt.pie(sizes, labels=labels)`
* Box: `plt.boxplot(data)`

### Customization

* Colors: `"r"`, `"g"`, `"b"`, or hex codes.
* Markers: `"o"`, `"s"`, `"^"`.
* Subplots: `plt.subplot(rows, cols, index)`.
* Figure size: `plt.figure(figsize=(8,4))`
* Add grid: `plt.grid(True)`
* Save figure: `plt.savefig('plot.png')`

### Integration with pandas

* Plot directly from DataFrames: `df.plot()`

### Applications in Data Science

* Exploratory Data Analysis (EDA): visualize distributions, trends, and relationships.
* Model evaluation: plot predictions vs. actual values, confusion matrices, ROC curves.
* Publication-quality figures: customize fonts, colors, and layouts; export to PNG, PDF, SVG.

---

## 6. Python Dictionaries

* **Definition:** Unordered, mutable collections of key-value pairs.
* Keys must be unique & immutable; values can be any type.

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

* Data mapping (e.g., `{ "M": 1, "F": 0 }`).
* Counting with `collections.Counter`.
* JSON-like data structures.
* Configurations and settings storage.

---

## 7. Pandas

* **Pandas** is a powerful Python library for data manipulation and analysis.
* Provides two main data structures:
  - **Series:** 1-dimensional labeled array.
  - **DataFrame:** 2-dimensional labeled table (rows and columns).

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
* Info: `df.info()`, `df.describe()`
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

## Comparison Operators in Python

* **Comparison operators** are used to compare values and return a Boolean result (`True` or `False`).

| Operator | Description           | Example         | Result      |
|----------|----------------------|-----------------|-------------|
| `==`     | Equal to             | `5 == 5`        | `True`      |
| `!=`     | Not equal to         | `5 != 3`        | `True`      |
| `>`      | Greater than         | `7 > 2`         | `True`      |
| `<`      | Less than            | `3 < 8`         | `True`      |
| `>=`     | Greater than or equal| `6 >= 6`        | `True`      |
| `<=`     | Less than or equal   | `4 <= 5`        | `True`      |

### Usage Example

```python
x = 10
y = 5

print(x > y)      # True
print(x == y)     # False
print(x != y)     # True
```

* Commonly used in conditional statements (`if`, `elif`, `while`) to control

