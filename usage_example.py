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