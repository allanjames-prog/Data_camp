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

## For Loops in Data Science

* **For loops** are used to iterate over sequences (lists, arrays, DataFrame rows, etc.).
* Essential for processing datasets, applying transformations, aggregating results, and tracking element positions.

### Syntax

```python
for item in sequence:
    # code block
```

### Example: Iterating Over a List

```python
data = [5, 7, 3]
for value in data:
    print(value)
# Output: 5 7 3
```

### Example: Applying a Transformation

```python
squared = []
for value in data:
    squared.append(value ** 2)
print(squared)  # Output: [25, 49, 9]
```

### Example: Iterating Over a pandas DataFrame

```python
import pandas as pd
df = pd.DataFrame({'Age': [25, 30, 22]})
for index, row in df.iterrows():
    print(row['Age'])
# Output: 25 30 22
```

### Example: Printing Index and Value with enumerate

```python
data = [5, 7, 3]
for index, value in enumerate(data):
    print(f"Index: {index}, Value: {value}")
# Output:
# Index: 0, Value: 5
# Index: 1, Value: 7
# Index: 2, Value: 3
```

### Example: List Comprehension (Concise Alternative)

```python
squared = [value ** 2 for value in data]
print(squared)  # Output: [25, 49, 9]
```

### Notes

* For loops are commonly used for feature engineering, data cleaning, and aggregating statistics.
* `enumerate()` is useful for accessing both index and value, which helps in tracking positions or updating elements.
* List comprehensions can often replace for loops for more concise code.

---

## Iterating Over Dictionaries with `.items()`

* Use `.items()` to loop through key-value pairs in a dictionary.
* Useful for mapping, encoding, and summarizing categorical data in data science.

```python
person = {"name": "James", "age": 25, "city": "Kampala"}
for key, value in person.items():
    print(f"{key}: {value}")
# Output:
# name: James
# age: 25
# city: Kampala
```

---

## NumPy Arrays: Handling Multi-Dimensional Data

* NumPy arrays (`ndarray`) can be 1D, 2D, or higher dimensions (matrices, tensors).
* Used for storing and manipulating large datasets, images, time series, etc.

```python
import numpy as np

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d.shape)  # Output: (2, 3)
print(arr_2d[0, 1])  # Access element at row 0, column 1 (Output: 2)
```

* Slicing and indexing:
  ```python
  print(arr_2d[:, 1])  # All rows, column 1
  print(arr_2d[1, :])  # Row 1, all columns
  ```

---

## Vectorized Operations in NumPy

* **Vectorization** means applying operations to entire arrays, not element by element.
* Replaces explicit loops, making code faster and more readable.

```python
arr = np.array([1, 2, 3, 4])
result = arr * 2           # Multiplies every element by 2
sum_arr = arr + np.array([10, 20, 30, 40])  # Element-wise addition
```

* Aggregations:
  ```python
  mean = np.mean(arr)      # Mean of all elements
  std = np.std(arr)        # Standard deviation
  ```

* Boolean indexing:
  ```python
  filtered = arr[arr > 2]  # Select elements greater than 2
  ```

* **Benefits for Data Science:**  
  - Faster computations on large datasets.
  - Cleaner, more concise code.
  - Essential for machine learning, statistics, and data preprocessing.

---

### Looping Over a pandas DataFrame

* You can iterate over DataFrame rows to process or analyze data row by row.
* Common methods: `.iterrows()`, `.itertuples()`

#### Using `.iterrows()`

```python
import pandas as pd
df = pd.DataFrame({'Name': ['Ali', 'James', 'Sara'], 'Age': [25, 30, 22]})

for index, row in df.iterrows():
    print(f"Index: {index}, Name: {row['Name']}, Age: {row['Age']}")
# Output:
# Index: 0, Name: Ali, Age: 25
# Index: 1, Name: James, Age: 30
# Index: 2, Name: Sara, Age: 22
```

#### Using `.itertuples()` (more efficient)

```python
for row in df.itertuples():
    print(f"Index: {row.Index}, Name: {row.Name}, Age: {row.Age}")
```

* Iterating is useful for custom calculations, feature engineering, or exporting data.
* For large datasets, prefer vectorized operations for better performance.

---

## Random Numbers in Python for Data Science

Random numbers are widely used in data science for tasks such as sampling, simulation, shuffling data, initializing model parameters, and creating synthetic datasets.

### Why Use Random Numbers?

- **Splitting datasets:** Randomly divide data into training and testing sets.
- **Sampling:** Select random samples for experiments, bootstrapping, or cross-validation.
- **Simulation:** Generate synthetic data to test algorithms or models.
- **Machine learning:** Initialize weights and parameters randomly for algorithms.

### Generating Random Numbers with NumPy

```python
import numpy as np

# Random floats between 0 and 1
random_floats = np.random.rand(5)  # 1D array of 5 random floats

# Random integers in a range
random_ints = np.random.randint(10, 21, size=5)  # 5 integers from 10 to 20

# Random sample from an array (without replacement)
arr = np.arange(100)
sample = np.random.choice(arr, size=10, replace=False)

# Random numbers from a normal distribution (mean=0, std=1)
normal_data = np.random.randn(100)
```

### Setting a Seed for Reproducibility

A **seed** is a fixed starting point for the random number generator. Setting a seed ensures that random numbers are the same every time you run your code, which is crucial for reproducible experiments and sharing results.

```python
np.random.seed(42)
print(np.random.rand(3))  # Will always print the same numbers for seed 42
```

### Random Numbers with Python's `random` Module

```python
import random

# Random float between 0 and 1
print(random.random())

# Random integer between 1 and 10
print(random.randint(1, 10))

# Shuffle a list
data = [1, 2, 3, 4, 5]
random.shuffle(data)
print(data)
```

### Applications in Data Science

- **Data splitting:** `train_test_split` in scikit-learn uses random numbers to shuffle and split data.
- **Bootstrapping:** Randomly resample data to estimate statistics.
- **Monte Carlo simulations:** Use random numbers to model uncertainty and variability.
- **Synthetic data generation:** Create random datasets for testing and validation.

### Best Practices

- Always set a seed (`np.random.seed()`) when you need reproducible results.
- Use NumPy for large-scale random number generation and array operations.
- Use Python's `random` module for simple tasks with lists and small datasets.

---


