
---

# KMeansPlus by Lakshay jain

An end-to-end, class-based implementation of K-Means clustering written in pure Python and NumPy.
Unlike most “from scratch” tutorials, this project is designed to feel closer to a real tool (like scikit-learn) rather than a quick demo script.

It includes robust initialization, convergence checks, empty cluster handling, and even a demonstration of the **elbow method** for choosing the optimal number of clusters.

---

## Features

* `fit_predict` method
* Convergence detection (stops when cluster assignments stabilize)
* Multiple restarts (`n_init`) to avoid unlucky seeds
* Empty cluster handling (no silent NaNs)
* Manhattan distance updates (more robust against outliers)
* **K-Means++ initialization** for better centroid seeding
* Example notebook with the **elbow method** to find the best `k`

---

## Installation

This project is lightweight and only requires NumPy pandas.

```bash
pip install numpy pandas
```

Clone the repo and open the notebook to get started:

```bash
git clone https://github.com/lakshay-jainn/kmeans-plus-scratch.git
cd kmeans-plus-scratch
```

---

## Usage

Here’s a minimal example:

```python
import pandas as pd
from kmeans_plus import KMeansPlus

# Example dataset
df = pd.DataFrame({
    "x": [1, 2, 3, 8, 9, 10],
    "y": [1, 2, 3, 8, 9, 10]
})

# Initialize model
kmeans = KMeansPlus(k=2, n_init=10, iters=100)

# Fit and predict
clusters, points, labels, inertia = kmeans.fit_predict(df)

print("Cluster centers:", clusters)
print("Labels:", labels)
```

---

## Elbow Method Demonstration

The notebook `final.ipynb` includes a full demonstration of the **elbow method**, showing how to compute inertia values across different `k` and visualize the “elbow” point to guide the choice of optimal clusters.

---

## Why This Project?

Most “from scratch” implementations only explain the math.
This project goes further: it’s written as a reusable class, handles real-world edge cases, and mimics the design choices of libraries like scikit-learn.

The goal was to learn clustering more deeply by building something that feels usable, not just demonstrative.

---

## License

This project is open source and available under the MIT License.

---
