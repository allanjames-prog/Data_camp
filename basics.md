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
* Figure size: `plt.figure(figsize=(8,4))`.

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

### Applications

* Data mapping (e.g., `{ "M": 1, "F": 0 }`).
* Counting with `collections.Counter`.
* JSON-like data structures.
* Configurations and settings storage.

---

