import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv("Mall_Customers.csv")

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

numeric_data = data.drop(['CustomerID', 'Gender'], axis=1)

scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(numeric_data)

wcss = []
for i in range(1, len(data) + 1):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, len(data) + 1), wcss)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method to Determine Optimal k")
plt.show()

optimal_k = 5 ## choose from the elbow method

kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(x_scaled)

data['cluster_pred'] = kmeans.labels_

sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)", hue='cluster_pred', palette='rainbow')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"K-means Clustering (k={optimal_k})")
plt.show()
