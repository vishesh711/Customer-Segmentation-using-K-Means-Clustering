# Customer Segmentation Using K-Means Clustering

## Project Overview

This project applies the K-Means clustering algorithm to segment customers based on their annual income and spending score. The goal is to identify distinct customer segments within a dataset of mall customers, which can be used for targeted marketing.

## Prerequisites

Before running the code, ensure you have the following libraries installed:

- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used is `Mall_Customers.csv`, which contains information on 200 customers, including:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars
- **Spending Score (1-100)**: Score assigned by the mall based on customer behavior and spending nature

## Code Explanation

### 1. Importing Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

### 2. Loading the Dataset

```python
customer_data = pd.read_csv('/content/Mall_Customers.csv')
```

### 3. Data Exploration

- **Viewing the First Few Records:**

    ```python
    customer_data.head()
    ```

- **Dataset Information:**

    ```python
    customer_data.info()
    ```

- **Statistical Summary:**

    ```python
    customer_data.describe()
    ```

- **Checking for Missing Values:**

    ```python
    customer_data.isnull().sum()
    ```

### 4. Data Preprocessing

Extracting the relevant features (`Annual Income (k$)` and `Spending Score (1-100)`) for clustering:

```python
x = customer_data.iloc[:, [3, 4]].values
```

### 5. Determining the Optimal Number of Clusters

Using the Elbow Method to determine the optimal number of clusters:

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

### 6. Applying K-Means Clustering

```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(x)
```

### 7. Visualizing the Clusters

```python
plt.figure(figsize=(8, 8))
plt.scatter(x[Y == 0, 0], x[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(x[Y == 1, 0], x[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(x[Y == 2, 0], x[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[Y == 3, 0], x[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(x[Y == 4, 0], x[Y == 4, 1], s=50, c='blue', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

## Conclusion

The K-Means algorithm successfully segments customers into five distinct groups based on their annual income and spending score. These clusters can be used by businesses to understand customer behavior and tailor marketing strategies accordingly.
