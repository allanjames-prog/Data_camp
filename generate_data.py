import pandas as pd
import numpy as np

# Set number of rows
num_rows = 1000  # you can increase this (e.g., 5000 or 10000)

# Generate random data
data = {
    "id": range(1, num_rows + 1),
    "age": np.random.randint(18, 60, size=num_rows),
    "salary": np.random.randint(50000, 200000, size=num_rows),
    "department": np.random.choice(["IT", "HR", "Sales", "Finance"], size=num_rows),
    "years_experience": np.random.randint(0, 20, size=num_rows),
    "performance_score": np.round(np.random.uniform(1.0, 5.0, size=num_rows), 1)
}

# Create and save DataFrame
df = pd.DataFrame(data)
df.to_csv("random_data.csv", index=False)

print(f"Random CSV with {num_rows} rows generated successfully!")
