Here are concise notes on each topic:

## Python Basics
- Python is an interpreted, high-level, general-purpose programming language.
- Variables store data: `x = 5`
- Data types: `int`, `float`, `str`, `bool`
- Basic operators: `+`, `-`, `*`, `/`, `%`
- Control flow: `if`, `elif`, `else`, `for`, `while`
- Indentation defines code blocks.

## Python Lists
- Lists store ordered, mutable collections: `my_list = [1, 2, 3]`
- Access elements: `my_list[0]`
- Modify: `my_list.append(4)`, `my_list.remove(2)`
- Slicing: `my_list[1:3]`
- Lists can hold mixed data types.

## Functions and Packages
- Functions: reusable blocks of code.
    ```python
    def add(a, b):
        return a + b
    ```
- Call functions: `add(2, 3)`
- Packages: collections of modules. Import with `import package_name`
- Use functions from packages: `math.sqrt(4)`

## NumPy
- NumPy is a package for numerical computing.
- Provides `ndarray` for efficient array operations.
    ```python
    import numpy as np
    arr = np.array([1, 2, 3])
    ```
- Supports vectorized operations: `arr * 2`
- Useful functions: `np.mean(arr)`, `np.arange(0, 10, 2)`
- Commonly used for data science and machine learning.

Here are detailed notes on NumPy and its common operations in data science:

## NumPy Overview
- NumPy (Numerical Python) is a core library for scientific computing in Python.
- Provides the `ndarray` object for fast, memory-efficient array operations.
- Supports mathematical, logical, and statistical operations on arrays.

## Creating Arrays
```python
import numpy as np

# From a Python list
arr = np.array([1, 2, 3])

# Zeros, ones, random
zeros = np.zeros((2, 3))
ones = np.ones(5)
rand = np.random.rand(3, 2)

# Range and reshape
arange = np.arange(0, 10, 2)
reshaped = arange.reshape((5, 1))
```

## Array Operations
- **Element-wise operations:** `arr * 2`, `arr + 5`
- **Arithmetic:** `np.add(arr1, arr2)`, `np.subtract(arr1, arr2)`
- **Aggregation:** `np.sum(arr)`, `np.mean(arr)`, `np.std(arr)`, `np.min(arr)`, `np.max(arr)`
- **Matrix multiplication:** `np.dot(arr1, arr2)` or `arr1 @ arr2`
- **Transposing:** `arr.T`
- **Indexing and slicing:** `arr[0:3]`, `arr[:, 1]`

## Useful Functions
- **Sorting:** `np.sort(arr)`
- **Unique values:** `np.unique(arr)`
- **Concatenation:** `np.concatenate([arr1, arr2])`
- **Stacking:** `np.vstack([arr1, arr2])`, `np.hstack([arr1, arr2])`
- **Filtering:** `arr[arr > 5]`

## Data Science Applications
- **Data preprocessing:** Cleaning, normalizing, and transforming data.
- **Statistical analysis:** Calculating mean, median, standard deviation, correlations.
- **Linear algebra:** Matrix operations for machine learning algorithms.
- **Handling missing data:** `np.isnan(arr)` to find NaNs, `np.nanmean(arr)` to ignore NaNs in calculations.
- **Random sampling:** `np.random.choice(arr, size=5)`

## Example: Basic Data Analysis
```python
import numpy as np

data = np.array([10, 20, 30, 40, 50])
mean = np.mean(data)
std_dev = np.std(data)
filtered = data[data > 25]  # Filter values greater than 25
```
NumPy is essential for efficient numerical computations and is widely used with pandas, scikit-learn, and other data science libraries.


## Matplotlib (Detailed Notes)

- **Matplotlib** is a comprehensive Python library for creating static, interactive, and animated visualizations.
- Most commonly used module: `pyplot` (`import matplotlib.pyplot as plt`), which provides a simple interface for plotting.

### Basic Plotting
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y, label='Line Plot')   # Create a line plot
plt.xlabel('X-axis')                # Label x-axis
plt.ylabel('Y-axis')                # Label y-axis
plt.title('Simple Line Plot')       # Add a title
plt.legend()                        # Show legend
plt.grid(True)                      # Add grid
plt.show()                          # Display plot
```

### Common Plot Types
- **Line plot:** `plt.plot(x, y)` — visualize trends over time or ordered categories.
- **Scatter plot:** `plt.scatter(x, y)` — show relationships or distributions between two variables.
- **Bar plot:** `plt.bar(x, y)` — compare quantities across categories.
- **Histogram:** `plt.hist(data, bins=10)` — display the distribution of numerical data.
- **Pie chart:** `plt.pie(sizes, labels=labels)` — show proportions of categories.
- **Box plot:** `plt.boxplot(data)` — visualize data spread and outliers.

### Customization
- **Colors, markers, line styles:**  
  `plt.plot(x, y, color='red', marker='o', linestyle='--')`
- **Legends:**  
  `plt.legend(['Series 1'])`
- **Grid:**  
  `plt.grid(True)`
- **Figure size:**  
  `plt.figure(figsize=(8, 4))`
- **Subplots:**  
  ```python
  plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
  plt.plot(x, y)
  plt.subplot(1, 2, 2)  # 2nd plot
  plt.bar(x, y)
  plt.show()
  ```

### Data Science Applications
- **Exploratory Data Analysis (EDA):**  
  - Visualize distributions (histograms, boxplots)
  - Detect trends and patterns (line plots, scatter plots)
  - Compare categories (bar plots, pie charts)
- **Model Evaluation:**  
  - Plot predictions vs. actual values
  - Visualize confusion matrices and ROC curves
- **Publication-quality figures:**  
  - Customize fonts, colors, and layouts for reports and presentations
  - Export plots to formats like PNG, PDF, SVG

- Matplotlib integrates well with NumPy and pandas, making it a key tool for data


