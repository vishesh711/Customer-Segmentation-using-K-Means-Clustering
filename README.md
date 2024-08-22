Customer Segmentation using K-Means Clustering
Overview
This project demonstrates the use of the K-Means clustering algorithm to segment customers based on their annual income and spending score. The goal is to identify distinct customer groups within a dataset of mall customers.

Libraries Required
NumPy: For numerical operations.
Pandas: For data manipulation and analysis.
Matplotlib: For plotting graphs and visualizing data.
Seaborn: For statistical data visualization.
Scikit-learn (sklearn): For machine learning algorithms, specifically K-Means clustering.
Steps Performed
1. Importing Libraries
python
Copy code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
2. Loading the Dataset
The dataset Mall_Customers.csv is loaded into a pandas DataFrame.

python
Copy code
customer_data = pd.read_csv('/content/Mall_Customers.csv')
3. Exploring the Dataset
customer_data.head(): Displays the first five rows of the dataset.
customer_data.info(): Provides information about the dataset, including the number of entries, data types, and non-null counts.
customer_data.describe(): Provides summary statistics of the numerical columns.
customer_data.isnull().sum(): Checks for missing values in the dataset.
4. Data Preprocessing
The relevant features (Annual Income (k$) and Spending Score (1-100)) are extracted and converted into a NumPy array x.

python
Copy code
x = customer_data.iloc[:, [3, 4]].values
5. Determining the Optimal Number of Clusters
The Elbow Method is used to find the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against the number of clusters.

python
Copy code
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
6. Applying K-Means Clustering
The K-Means algorithm is applied to segment the customers into 5 clusters.

python
Copy code
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(x)
7. Visualizing the Clusters
The customer segments are visualized using a scatter plot, with different colors representing different clusters.

python
Copy code
plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0], x[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[Y==1,0], x[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(x[Y==2,0], x[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[Y==3,0], x[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(x[Y==4,0], x[Y==4,1], s=50, c='blue', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
Result
The code segments customers into five distinct groups based on their annual income and spending score, with the centroids of each cluster marked on the plot. This visualization helps in understanding the different customer segments and can be used for targeted marketing strategies.

